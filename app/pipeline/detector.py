"""Scoring layer.

Wraps the novelty detector trained by `scripts/train_model.py` and forms the
first stage of the funnel: it reduces a batch of thousands of events to a few
dozen candidates before a single token is spent on an LLM.

The exposed score is normalized by the threshold, so 0.5 is exactly the cutoff.
That keeps the correlator and the API independent of the model's
log-likelihood scale — swapping the detector requires no recalibration
downstream.

The floor is recall-first. The project's criterion is zero false negatives: an
attack classified as benign is the worst possible error, while a false positive
merely creates triage work, which is what the agents do. Hence the default runs
at 0.2% FPR, where incident recall saturates, rather than at a stricter point
with better precision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib

from app.pipeline.novelty import NoveltyDetector
from app.pipeline.parser import LogEvent

logger = logging.getLogger(__name__)

# The normalized score places the threshold at 0.5 by construction.
NORMALIZED_THRESHOLD = 0.5

# An event up to 10% below the floor does not open an incident, but it is counted
# and reported. Under the zero-false-negative policy no batch may be returned as
# clean without stating how close the traffic came to the cutoff.
NEAR_THRESHOLD = 0.45

# Severity by distance to stricter operating points, read from the calibration
# table. Saying "this event would still be flagged at 0.02% FPR" is a measured
# claim; saying "score >= 0.66 is severe" was a constant that silently drifted
# every time the operating point moved.
SEVERITY_FPR = (("high", 0.0002), ("medium", 0.001))


class ModelNotLoaded(RuntimeError):
    pass


class Detector:
    """Loads `detector.joblib` and scores events.

    A failed load does not bring the process down: `is_loaded` stays false,
    /health reports `degraded` and /analyze returns 503. A missing model is an
    operational error, not a request error.
    """

    def __init__(self, model_path: Path | str, target_fpr: float | None = None) -> None:
        self.model_path = Path(model_path)
        self.novelty: NoveltyDetector | None = None
        self.metadata: dict = {}
        self.load_error: str | None = None
        self.applied_fpr: float | None = None
        self._severity_cutoffs: list[tuple[str, float]] = []
        self._load(target_fpr)

    def _load(self, target_fpr: float | None) -> None:
        if not self.model_path.exists():
            self.load_error = f"artifact not found at {self.model_path}"
            logger.warning("%s — run scripts/train_model.py", self.load_error)
            return
        try:
            artifact = joblib.load(self.model_path)
            self.novelty = artifact["detector"]
            self.metadata = artifact.get("metadata", {})
        except Exception as exc:  # pragma: no cover - I/O failure path
            self.load_error = f"failed to load model: {exc}"
            logger.exception(self.load_error)
            return

        if target_fpr is not None and self.novelty.calibration:
            self.applied_fpr = self.novelty.calibrate_to(target_fpr)
            if self.applied_fpr != target_fpr:
                logger.warning(
                    "target FPR %.4f is absent from the calibration table; using %.4f",
                    target_fpr, self.applied_fpr,
                )
            logger.info(
                "detector calibrated to %.2f%% FPR (threshold %.4f)",
                self.applied_fpr * 100, self.novelty.url_threshold,
            )

        self._severity_cutoffs = [
            (label, self.novelty.normalized_for_fpr(rate)) for label, rate in SEVERITY_FPR
        ]

    @property
    def is_loaded(self) -> bool:
        return self.novelty is not None

    @property
    def event_threshold(self) -> float:
        return NORMALIZED_THRESHOLD

    def _require(self) -> NoveltyDetector:
        if self.novelty is None:
            raise ModelNotLoaded(self.load_error or "model not loaded")
        return self.novelty

    def score_events(self, events: Sequence[LogEvent]) -> list[float]:
        """Normalized improbability per event. 0.5 is the threshold."""
        novelty = self._require()
        return [novelty.normalized_score(event.decoded_url) for event in events]

    def explain(self, event: LogEvent) -> str:
        """The least likely substring of the URL.

        This is what turns the score into quotable evidence: the dossier shows the
        agent which span of the request does not look like normal traffic.
        """
        return self._require().explain(event.decoded_url)

    def severity(self, score: float) -> str:
        """Severity of an event derived from the measured operating points."""
        for label, cutoff in self._severity_cutoffs:
            if cutoff == cutoff and score >= cutoff:  # cutoff == cutoff excludes NaN
                return label
        return "low"

    def count_near_threshold(self, scores: Sequence[float]) -> int:
        """Events that landed just below the floor without opening an incident."""
        return sum(1 for score in scores if NEAR_THRESHOLD <= score < NORMALIZED_THRESHOLD)
