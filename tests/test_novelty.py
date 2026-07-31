"""Tests for the novelty model, without depending on the trained artifact.

The models here are fitted on the fly from a few dozen synthetic URLs. They do not
measure detection quality — that is `test_detector.py`'s job, which uses the real
artifact — but the structural properties the approach promises: length
normalization, absence of labels in the fit, and localization of the improbable
span.
"""

from __future__ import annotations

import pytest

from app.pipeline.novelty import NoveltyDetector, UrlNoveltyModel

NORMAL_URLS = [
    f"/product/{index}/detail" for index in range(40)
] + [
    f"/image/{index}/productModel/200x200" for index in range(40)
] + [
    f"/filter?f=p{index}&page=2" for index in range(20)
]

INJECTION = "/image/1?wh=50x50' UNION ALL SELECT NULL,NULL,NULL-- aBcD"


@pytest.fixture(scope="module")
def model() -> UrlNoveltyModel:
    return UrlNoveltyModel().fit(NORMAL_URLS)


class TestUrlNoveltyModel:
    def test_familiar_url_scores_below_an_injection(self, model):
        assert model.score("/product/7/detail") < model.score(INJECTION)

    def test_empty_url_scores_zero(self, model):
        assert model.score("") == 0.0

    def test_score_is_normalized_by_length(self, model):
        """Without normalization, repeating the normal pattern would raise the score."""
        short = model.score("/product/1/detail")
        long = model.score("/product/1/detail" * 6)
        assert abs(short - long) < 0.5

    def test_fit_receives_no_label(self, model):
        """`fit` accepts text only — there is no way to leak a label through that API."""
        import inspect

        signature = inspect.signature(UrlNoveltyModel.fit)
        assert list(signature.parameters) == ["self", "texts"]

    def test_explain_locates_the_improbable_span(self, model):
        span = model.explain(INJECTION)
        assert span
        assert span in INJECTION
        # The span must contain part of the payload, not the banal prefix.
        assert any(token in span.upper() for token in ("UNION", "SELECT", "NULL", "'"))

    def test_explain_of_short_text_returns_the_whole_text(self, model):
        assert model.explain("/abc") == "/abc"

    def test_explain_of_empty_text(self, model):
        assert model.explain("") == ""

    def test_freeze_removes_defaultdict(self):
        """A serialized defaultdict grows again on every missing lookup."""
        from collections import defaultdict

        frozen = UrlNoveltyModel().fit(NORMAL_URLS)
        assert not any(isinstance(counts, defaultdict) for counts in frozen.ngram_counts)
        assert not any(isinstance(counts, defaultdict) for counts in frozen.context_counts)

    def test_missing_key_lookup_does_not_grow_the_model(self, model):
        before = sum(len(counts) for counts in model.ngram_counts)
        model.score("/xyz-completely-unknown-cao")
        assert sum(len(counts) for counts in model.ngram_counts) == before

    def test_scoring_is_deterministic(self, model):
        assert model.score(INJECTION) == model.score(INJECTION)

    def test_memoization_does_not_change_the_result(self, model):
        fresh = UrlNoveltyModel().fit(NORMAL_URLS)
        assert fresh.score(INJECTION) == pytest.approx(model.score(INJECTION))


class TestNoveltyDetector:
    def _detector(self) -> NoveltyDetector:
        url_model = UrlNoveltyModel().fit(NORMAL_URLS)
        threshold = url_model.score("/product/1/detail") * 1.5
        return NoveltyDetector(url_model, threshold)

    def test_normalized_threshold_sits_at_one_half(self):
        """0.5 on the normalized scale is exactly the threshold, by construction.

        That is what decouples the rest of the system from the NLL scale: moving the
        operating point shifts no constant downstream.
        """
        detector = self._detector()
        reference = UrlNoveltyModel().fit(NORMAL_URLS)
        # Artificially set the threshold to the raw score of this very URL.
        detector.url_threshold = reference.score(INJECTION)
        assert detector.normalized_score(INJECTION) == pytest.approx(0.5)

    def test_normal_url_stays_below_the_threshold(self):
        assert self._detector().normalized_score("/product/3/detail") < 0.5

    def test_injection_goes_above_the_threshold(self):
        assert self._detector().normalized_score(INJECTION) > 0.5

    def test_normalized_score_is_capped(self):
        detector = self._detector()
        assert detector.normalized_score("!" * 300) < 1.0

    def test_explain_returns_a_span_of_the_url(self):
        detector = self._detector()
        assert detector.explain(INJECTION) in INJECTION

    def test_without_a_calibration_table_there_is_no_severity_band(self):
        """Severity derived from an absent calibration returns NaN, not a guess."""
        detector = self._detector()
        value = detector.normalized_for_fpr(0.001)
        assert value != value  # NaN
