from app.db import Database


def sample_report(incident_id: str = "inc_1", ip: str = "203.0.113.9") -> dict:
    return {
        "incident_id": incident_id,
        "ip": ip,
        "verdict": "malicious",
        "severity": "critical",
        "confidence": 0.95,
        "recommended_action": "block",
        "analyzed_by": "llm",
        "event_count": 6,
        "suspicious_count": 2,
        "max_event_score": 0.97,
        "summary": "SQL injection.",
        "actions": [
            {
                "type": "block", "target": ip, "reason": "sqli",
                "executed": False, "detail": "dry-run",
            }
        ],
    }


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    Database(path)
    Database(path).init_schema()


def test_saves_and_retrieves_an_analysis(database: Database):
    database.save_analysis("anl_1", "siem", 100, 98, 3, True, "completed", 1200)
    record = database.get_analysis_status("anl_1")
    assert record["events_received"] == 100
    assert record["threat_detected"] == 1
    assert record["duration_ms"] == 1200


def test_missing_analysis_returns_none(database: Database):
    assert database.get_analysis_status("anl_does_not_exist") is None


def test_saves_incident_and_actions(database: Database):
    database.save_analysis("anl_1", "siem", 6, 6, 2, True, "completed")
    database.save_incident("anl_1", sample_report())
    retrieved = database.get_incident("inc_1")
    assert retrieved["verdict"] == "malicious"
    assert retrieved["actions"][0]["type"] == "block"
    with database.connect() as connection:
        total = connection.execute("SELECT COUNT(*) AS n FROM actions").fetchone()["n"]
    assert total == 1


def test_missing_incident_returns_none(database: Database):
    assert database.get_incident("inc_does_not_exist") is None


def test_incident_list_filters_by_ip_and_analysis(database: Database):
    database.save_analysis("anl_1", "s", 1, 1, 1, True, "completed")
    database.save_analysis("anl_2", "s", 1, 1, 1, True, "completed")
    database.save_incident("anl_1", sample_report("inc_1", "1.1.1.1"))
    database.save_incident("anl_2", sample_report("inc_2", "2.2.2.2"))

    assert len(database.list_incidents()) == 2
    assert len(database.list_incidents(ip="1.1.1.1")) == 1
    assert len(database.list_incidents(analysis_id="anl_2")) == 1
    assert database.list_incidents(ip="1.1.1.1", analysis_id="anl_2") == []


def test_blocklist_is_idempotent(database: Database):
    database.add_to_blocklist("203.0.113.9", "sqli", "dry_run", "inc_1")
    database.add_to_blocklist("203.0.113.9", "repeat offender", "dry_run", "inc_2")
    entries = database.list_blocklist()
    assert len(entries) == 1
    assert entries[0]["reason"] == "repeat offender"
    assert database.is_blocked("203.0.113.9") is True
    assert database.is_blocked("10.0.0.1") is False


def test_ip_history_feeds_the_agent(database: Database):
    """History is an investigation tool, not only an audit trail."""
    database.save_analysis("anl_1", "s", 1, 1, 1, True, "completed")
    database.save_incident("anl_1", sample_report("inc_1", "203.0.113.9"))
    database.add_to_blocklist("203.0.113.9", "sqli", "dry_run", "inc_1")

    history = database.get_ip_history("203.0.113.9")
    assert history["previous_incidents"] == 1
    assert history["currently_blocked"] is True
    assert history["history"][0]["verdict"] == "malicious"


def test_history_of_unknown_ip_is_empty(database: Database):
    history = database.get_ip_history("198.51.100.1")
    assert history["previous_incidents"] == 0
    assert history["currently_blocked"] is False
