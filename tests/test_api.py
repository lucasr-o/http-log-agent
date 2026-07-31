import pytest
from fastapi.testclient import TestClient

from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.config import get_settings
from app.db import Database
from app.main import app
from app.orchestrator import Orchestrator
from tests.conftest import (
    VERDICT_MALICIOUS,
    FakeClient,
    StubDetector,
    make_event,
    tool_use,
)
from tests.test_orchestrator import as_payload

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Boot the real application and swap only the external dependencies.

    Routing, schema validation and authentication are genuinely exercised; only the
    trained model and the LLM client are replaced.
    """
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    with TestClient(app) as test_client:
        database = Database(tmp_path / "api.db")
        detector = StubDetector()
        app.state.database = database
        app.state.detector = detector
        app.state.orchestrator = Orchestrator(
            settings=get_settings(), detector=detector, database=database,
            blocker=IPBlocker("dry_run"), notifier=Notifier(), llm_client=None,
        )
        test_client.database = database
        yield test_client

    get_settings.cache_clear()


def payload(events, **extra):
    return {"events": [as_payload(event) for event in events], **extra}


class TestAuthentication:
    def test_no_key_is_401(self, client):
        assert client.post("/analyze", json={"events": []}).status_code == 401

    def test_wrong_key_is_401(self, client):
        response = client.post(
            "/analyze", json={"events": []}, headers={"X-API-Key": "wrong"}
        )
        assert response.status_code == 401

    def test_public_endpoints_need_no_key(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200


class TestHealth:
    def test_reports_component_state(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["llm_enabled"] is False
        assert body["block_mode"] == "dry_run"


class TestAnalyze:
    def test_empty_batch_is_422(self, client):
        response = client.post("/analyze", json={"events": []}, headers=HEADERS)
        assert response.status_code == 422

    def test_batch_over_the_limit_is_413(self, client, monkeypatch):
        """A batch endpoint with no cap is a denial-of-service vector."""
        monkeypatch.setenv("MAX_BATCH_SIZE", "2")
        get_settings.cache_clear()
        events = [make_event(offset=i) for i in range(3)]
        response = client.post("/analyze", json=payload(events), headers=HEADERS)
        assert response.status_code == 413
        assert "maximum" in response.json()["detail"]

    def test_benign_traffic_answers_without_a_threat(self, client):
        events = [make_event(url=f"/product/{i}", offset=i) for i in range(5)]
        body = client.post("/analyze", json=payload(events), headers=HEADERS).json()
        assert body["threat_detected"] is False
        assert body["incidents"] == []
        assert body["recommended_action"] == "allow"

    def test_attack_opens_an_incident(self, client, attack_events):
        response = client.post("/analyze", json=payload(attack_events), headers=HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["threat_detected"] is True
        assert len(body["incidents"]) == 1
        assert body["incidents"][0]["sample_events"]

    def test_raw_log_lines_are_accepted(self, client):
        line = (
            '203.0.113.5 - - [23/Jan/2019:03:00:00 +0000] '
            '"GET /x?q=1%20UNION%20SELECT%201 HTTP/1.1" 200 12 "-" "sqlmap/1.4"'
        )
        body = client.post(
            "/analyze", json={"events": [line]}, headers=HEADERS
        ).json()
        assert body["events_parsed"] == 1
        assert body["threat_detected"] is True

    def test_event_without_ip_is_422(self, client):
        response = client.post(
            "/analyze",
            json={"events": [{"url": "/x", "time": "2019-01-23 03:00:00+00:00"}]},
            headers=HEADERS,
        )
        assert response.status_code == 422


class TestQueries:
    def test_persisted_incident_is_retrievable(self, client, attack_events):
        analysis = client.post("/analyze", json=payload(attack_events), headers=HEADERS).json()
        incident_id = analysis["incidents"][0]["incident_id"]

        detail = client.get(f"/incidents/{incident_id}", headers=HEADERS)
        assert detail.status_code == 200
        assert detail.json()["incident_id"] == incident_id

    def test_missing_incident_is_404(self, client):
        assert client.get("/incidents/inc_nothing", headers=HEADERS).status_code == 404

    def test_list_filters_by_analysis(self, client, attack_events):
        analysis = client.post("/analyze", json=payload(attack_events), headers=HEADERS).json()
        listing = client.get(
            "/incidents", params={"analysis_id": analysis["analysis_id"]}, headers=HEADERS
        ).json()
        assert len(listing) == 1
        assert listing[0]["analysis_id"] == analysis["analysis_id"]

    def test_analysis_carries_its_incidents(self, client, attack_events):
        analysis = client.post("/analyze", json=payload(attack_events), headers=HEADERS).json()
        detail = client.get(
            f"/analyses/{analysis['analysis_id']}", headers=HEADERS
        ).json()
        assert detail["status"] == "completed"
        assert len(detail["incidents"]) == 1

    def test_missing_analysis_is_404(self, client):
        assert client.get("/analyses/anl_nothing", headers=HEADERS).status_code == 404

    def test_blocklist_reflects_the_agent_action(self, client, attack_events):
        client.app.state.orchestrator.database.add_to_blocklist(
            "203.0.113.9", "confirmed sqli", "dry_run", "inc_1"
        )
        listing = client.get("/blocklist", headers=HEADERS).json()
        assert listing[0]["ip"] == "203.0.113.9"
        assert listing[0]["mode"] == "dry_run"


class TestAsyncPath:
    def test_long_batch_returns_202(self, client, monkeypatch, attack_events):
        """An exhausted deadline returns 202 instead of holding the HTTP connection."""
        monkeypatch.setattr(
            client.app.state.orchestrator.settings, "sync_timeout_seconds", 0.0
        )
        events = [
            make_event(url="/x?q=1 UNION SELECT 1", ip=f"10.0.0.{octet}")
            for octet in range(1, 5)
        ]
        response = client.post("/analyze", json=payload(events), headers=HEADERS)
        assert response.status_code == 202
        assert response.json()["status"] == "processing"

    def test_pending_incidents_finish_in_the_background(self, client, monkeypatch):
        monkeypatch.setattr(
            client.app.state.orchestrator.settings, "sync_timeout_seconds", 0.0
        )
        events = [
            make_event(url="/x?q=1 UNION SELECT 1", ip=f"10.0.0.{octet}")
            for octet in range(1, 4)
        ]
        body = client.post("/analyze", json=payload(events), headers=HEADERS).json()
        # TestClient runs the BackgroundTasks before returning the response.
        detail = client.get(f"/analyses/{body['analysis_id']}", headers=HEADERS).json()
        assert detail["status"] == "completed"
        assert len(detail["incidents"]) == 3


class TestWithAgents:
    def test_incident_analyzed_by_the_llm(self, client, attack_events, tmp_path):
        llm = FakeClient(
            [
                tool_use("submit_verdict", VERDICT_MALICIOUS),
                tool_use("block_ip", {"reason": "sqli"}, block_id="tu_b"),
                tool_use(
                    "submit_action_plan",
                    {"final_action": "block", "rationale": "confirmed exploitation"},
                    block_id="tu_c",
                ),
            ]
        )
        state = client.app.state
        state.orchestrator = Orchestrator(
            settings=get_settings(), detector=state.detector, database=state.database,
            blocker=IPBlocker("dry_run"), notifier=Notifier(), llm_client=llm,
        )
        body = client.post("/analyze", json=payload(attack_events), headers=HEADERS).json()
        incident = body["incidents"][0]
        assert incident["analyzed_by"] == "llm"
        assert incident["verdict"] == "malicious"
        assert incident["reasoning"]
        assert client.get("/blocklist", headers=HEADERS).json()[0]["ip"] == incident["ip"]
