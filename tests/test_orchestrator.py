from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.orchestrator import Orchestrator, deterministic_verdict
from app.schemas import AnalyzeRequest
from tests.conftest import (
    VERDICT_MALICIOUS,
    FakeClient,
    StubDetector,
    make_event,
    text_response,
    tool_use,
)


def as_payload(event) -> dict:
    return {
        "ip": event.ip,
        "time": event.timestamp.isoformat(),
        "method": event.method,
        "url": event.url,
        "status": event.status,
        "size": event.size,
        "user_agent": event.user_agent,
    }


def build(database, settings, client=None) -> Orchestrator:
    return Orchestrator(
        settings=settings,
        detector=StubDetector(),
        database=database,
        blocker=IPBlocker("dry_run"),
        notifier=Notifier(),
        llm_client=client,
    )


class TestDeterministicVerdict:
    def test_high_score_raises_an_alert(self, attack_dossier):
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert verdict["recommended_action"] == "alert"

    def test_never_claims_malicious_without_an_llm(self, attack_dossier):
        """The score distributions overlap: benign max 0.711 vs attack median 0.585.
        The score alone does not support the claim."""
        attack_dossier.suspicious[0].score = 0.99
        verdict = deterministic_verdict(attack_dossier, StubDetector().severity)
        assert verdict["verdict"] == "suspicious"
        assert verdict["confidence"] <= 0.6

    def test_severity_comes_from_the_detector(self, attack_dossier):
        """The bands derive from the calibration table, not from constants."""
        severity_of = StubDetector().severity
        attack_dossier.suspicious = attack_dossier.suspicious[:1]
        for score, expected in ((0.52, "low"), (0.75, "medium"), (0.95, "high")):
            attack_dossier.suspicious[0].score = score
            assert deterministic_verdict(attack_dossier, severity_of)["severity"] == expected

    def test_summary_quotes_the_improbable_span(self, attack_dossier):
        """The deterministic verdict also has to point at concrete evidence."""
        evidence = attack_dossier.top_samples()[0].evidence
        assert evidence
        assert evidence in deterministic_verdict(attack_dossier)["summary"]

    def test_deterministic_verdict_admits_its_own_limit(self, attack_dossier):
        reasoning = deterministic_verdict(attack_dossier)["reasoning"]
        assert "does not distinguish an attack from" in reasoning
        assert "does not issue benign verdicts" in reasoning


class TestAnalyze:
    def test_benign_batch_opens_no_incident(self, database, settings):
        events = [make_event(url=f"/product/{i}", offset=i) for i in range(10)]
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, pending = build(database, settings).analyze(request)

        assert response.threat_detected is False
        assert response.incidents == []
        assert response.recommended_action == "allow"
        assert pending == []

    def test_batch_with_an_attack_opens_an_incident(self, database, settings, attack_events):
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events])
        response, _ = build(database, settings).analyze(request)

        assert response.threat_detected is True
        assert len(response.incidents) == 1
        assert response.incidents[0].analyzed_by == "deterministic"
        assert database.get_incident(response.incidents[0].incident_id) is not None

    def test_invalid_events_are_reported(self, database, settings):
        """Pydantic blocks missing fields; what passes it but fails normalization is rejected."""
        request = AnalyzeRequest(
            events=[{"ip": "1.2.3.4", "url": "/x", "time": "yesterday morning"}]
        )
        response, _ = build(database, settings).analyze(request)
        assert response.events_parsed == 0
        assert len(response.rejected) == 1
        assert "timestamp" in response.rejected[0].reason

    def test_analysis_is_persisted(self, database, settings, attack_events):
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events], source="siem")
        response, _ = build(database, settings).analyze(request)
        record = database.get_analysis_status(response.analysis_id)
        assert record["source"] == "siem"
        assert record["status"] == "completed"

    def test_exhausted_deadline_leaves_incidents_pending(self, database, settings):
        """A large batch returns 202 instead of holding the connection."""
        events = []
        for octet in range(1, 6):
            events.append(make_event(url="/x?q=1 UNION SELECT 1", ip=f"10.0.1.{octet}"))
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, pending = build(database, settings).analyze(request, deadline_seconds=0)

        assert response.status == "processing"
        assert len(pending) == 4
        assert "processing" in response.message

    def test_process_pending_completes_the_analysis(self, database, settings):
        events = [
            make_event(url="/x?q=1 UNION SELECT 1", ip=f"10.0.0.{octet}")
            for octet in range(1, 4)
        ]
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        orchestrator = build(database, settings)
        response, pending = orchestrator.analyze(request, deadline_seconds=0)

        orchestrator.process_pending(response.analysis_id, pending, dry_run=False)
        assert database.get_analysis_status(response.analysis_id)["status"] == "completed"
        assert len(database.list_incidents(analysis_id=response.analysis_id)) == 3


class TestAgentIntegration:
    def test_full_path_through_both_agents(self, database, settings, attack_events):
        client = FakeClient(
            [
                tool_use("submit_verdict", VERDICT_MALICIOUS),
                tool_use("block_ip", {"reason": "confirmed sqli"}, block_id="tu_b"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "block", "rationale": "confirmed exploitation"},
                    block_id="tu_c",
                ),
            ]
        )
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events])
        response, _ = build(database, settings, client).analyze(request)

        incident = response.incidents[0]
        assert incident.analyzed_by == "llm"
        assert incident.verdict == "malicious"
        assert incident.mitre_techniques == ["T1190"]
        assert incident.recommended_action == "block"
        assert database.is_blocked(incident.ip)

    def test_benign_verdict_skips_the_action_agent(
        self, database, settings, attack_events
    ):
        """Real saving: a benign incident spends one call, not two."""
        benign = {**VERDICT_MALICIOUS, "verdict": "benign", "recommended_action": "allow"}
        client = FakeClient([tool_use("submit_verdict", benign)])
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events])
        response, _ = build(database, settings, client).analyze(request)

        assert len(client.messages.calls) == 1
        assert response.incidents[0].recommended_action == "allow"

    def test_agent_failure_falls_back_to_deterministic(
        self, database, settings, attack_events
    ):
        """An LLM outage degrades the verdict, not availability."""
        client = FakeClient([text_response("I cannot tell")])
        request = AnalyzeRequest(events=[as_payload(e) for e in attack_events])
        response, _ = build(database, settings, client).analyze(request)

        assert response.incidents[0].analyzed_by == "deterministic"
        assert response.threat_detected is True

    def test_call_budget_limits_how_many_incidents_reach_the_llm(
        self, database, settings
    ):
        settings.llm_call_budget = 1
        events = [
            make_event(url="/x?q=1 UNION SELECT 1", ip=f"10.0.0.{octet}")
            for octet in range(1, 4)
        ]
        benign = {**VERDICT_MALICIOUS, "verdict": "benign", "recommended_action": "allow"}
        client = FakeClient([tool_use("submit_verdict", benign)])
        request = AnalyzeRequest(events=[as_payload(e) for e in events])
        response, _ = build(database, settings, client).analyze(request)

        sources = [incident.analyzed_by for incident in response.incidents]
        assert sources.count("llm") == 1
        assert sources.count("deterministic") == 2

    def test_request_level_dry_run_prevents_blocking(self, database, settings, attack_events):
        client = FakeClient(
            [
                tool_use("submit_verdict", VERDICT_MALICIOUS),
                tool_use("block_ip", {"reason": "sqli"}, block_id="tu_b"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "block", "rationale": "x"},
                    block_id="tu_c",
                ),
            ]
        )
        request = AnalyzeRequest(
            events=[as_payload(e) for e in attack_events], dry_run=True
        )
        response, _ = build(database, settings, client).analyze(request)
        assert database.is_blocked(response.incidents[0].ip) is False
