"""Tests for the configurable operating point.

The detector is calibrated during training for a target false positive rate, but
the right choice depends on how much triage work the operation tolerates — which is
not a decision for whoever trains the model. The calibration table travels inside
the artifact so moving the operating point requires no retraining.

The motivation is concrete: the remaining false negatives are `UNION ALL SELECT`
payloads scoring just below the cutoff. That is a calibration-margin failure, not a
capability failure, and whoever operates the service needs to be able to fix it
without retraining.
"""

from pathlib import Path

import pytest

from app.pipeline.detector import Detector
from app.pipeline.novelty import NoveltyDetector, UrlNoveltyModel

MODEL_PATH = Path("models/detector.joblib")

CALIBRATION = {0.001: 5.0, 0.002: 4.5, 0.01: 4.0}


def build(calibration=CALIBRATION) -> NoveltyDetector:
    url_model = UrlNoveltyModel().fit(["/product/1", "/product/2", "/image/3"])
    return NoveltyDetector(url_model, 5.0, calibration)


class TestCalibrateTo:
    def test_moves_the_threshold_to_the_target(self):
        detector = build()
        assert detector.calibrate_to(0.01) == 0.01
        assert detector.url_threshold == 4.0

    def test_a_more_permissive_threshold_detects_more(self):
        detector = build()
        detector.calibrate_to(0.001)
        strict = detector.url_threshold
        detector.calibrate_to(0.01)
        assert detector.url_threshold < strict

    def test_target_outside_the_table_uses_the_nearest(self):
        detector = build()
        assert detector.calibrate_to(0.0011) == 0.001
        assert detector.url_threshold == 5.0

    def test_without_a_table_nothing_changes(self):
        detector = build(calibration={})
        before = detector.url_threshold
        assert detector.calibrate_to(0.01) != detector.calibrate_to(0.01)  # nan != nan
        assert detector.url_threshold == before

    def test_severity_band_comes_from_the_table(self):
        """Severity is measured, not constant: where another point's threshold falls."""
        detector = build()
        detector.calibrate_to(0.01)  # threshold 4.0
        # The 0.001 threshold (5.0) sits at 5.0 / 4.0 / 2 = 0.625 on the current scale.
        assert detector.normalized_for_fpr(0.001) == pytest.approx(0.625)
        # A more permissive point falls below 0.5, as expected.
        assert detector.normalized_for_fpr(0.01) == pytest.approx(0.5)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="artifact missing")
class TestRealArtifact:
    def test_artifact_carries_the_calibration_table(self):
        detector = Detector(MODEL_PATH)
        assert detector.novelty.calibration
        assert 0.002 in detector.novelty.calibration

    def test_applied_fpr_is_recorded(self):
        detector = Detector(MODEL_PATH, target_fpr=0.002)
        assert detector.applied_fpr == 0.002

    def test_severity_scales_with_the_operating_point(self):
        """The bands move together with the threshold instead of drifting out of date.

        That was the defect of the previous version: `score >= 0.66` was calibrated for
        the 0.1% FPR threshold and drifted silently whenever the operating point moved.
        """
        detector = Detector(MODEL_PATH, target_fpr=0.002)
        high = detector.novelty.normalized_for_fpr(0.0002)
        medium = detector.novelty.normalized_for_fpr(0.001)
        assert high > medium > 0.5
        assert detector.severity(high) == "high"
        assert detector.severity(medium) == "medium"
        assert detector.severity(0.51) == "low"

    def test_a_more_permissive_operating_point_catches_what_escaped(self):
        """The remaining false negatives sit right against the threshold.

        The payload below is textbook SQLi and escapes by a margin at the default
        point — 0.4924 against a 0.5 cutoff. Loosening the operating point recovers it
        without retraining. It is a margin failure, not a capability failure, and it is
        why the calibration table travels inside the artifact.
        """
        from tests.conftest import make_event

        event = make_event(
            url="/image/{{basketItem.id}}?type=productModel UNION ALL SELECT NULL#&wh=50x50"
        )
        default = Detector(MODEL_PATH, target_fpr=0.002)
        permissive = Detector(MODEL_PATH, target_fpr=0.005)

        assert default.score_events([event])[0] < default.event_threshold
        assert permissive.score_events([event])[0] >= permissive.event_threshold

    def test_the_quoted_variant_is_already_caught_at_the_default(self):
        """The same payload with the escape quote rises above the cutoff at 0.002.

        It contrasts with the previous test: what separates detected from escaped here
        is a single improbable character, not the attack family.
        """
        from tests.conftest import make_event

        event = make_event(
            url="/image/{{basketItem.id}}?type=productModel' UNION ALL SELECT NULL,NULL#"
            "&wh=50x50"
        )
        default = Detector(MODEL_PATH, target_fpr=0.002)
        assert default.score_events([event])[0] >= default.event_threshold
