"""SQLite persistence.

Stores the history of analyses, incidents, actions and the blocklist. The history
is not decorative: the triage agent queries an IP's previous analyses through the
`get_ip_history` tool, so persistence is part of the reasoning and not only of the
audit trail.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id      TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    events_received  INTEGER NOT NULL,
    events_parsed    INTEGER NOT NULL,
    events_suspicious INTEGER NOT NULL,
    threat_detected  INTEGER NOT NULL,
    status           TEXT NOT NULL,
    duration_ms      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id      TEXT PRIMARY KEY,
    analysis_id      TEXT NOT NULL,
    ip               TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    severity         TEXT NOT NULL,
    confidence       REAL NOT NULL,
    recommended_action TEXT NOT NULL,
    analyzed_by      TEXT NOT NULL,
    event_count      INTEGER NOT NULL,
    suspicious_count INTEGER NOT NULL,
    max_event_score  REAL NOT NULL,
    payload          TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
);

CREATE TABLE IF NOT EXISTS actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL,
    type         TEXT NOT NULL,
    target       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    executed     INTEGER NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS blocklist (
    ip           TEXT PRIMARY KEY,
    reason       TEXT NOT NULL,
    mode         TEXT NOT NULL,
    incident_id  TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_incidents_ip ON incidents (ip);
CREATE INDEX IF NOT EXISTS idx_incidents_analysis ON incidents (analysis_id);
CREATE INDEX IF NOT EXISTS idx_actions_incident ON actions (incident_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shared = sqlite3.connect(":memory:") if str(self.path) == ":memory:" else None
        if self._shared is not None:
            self._shared.row_factory = sqlite3.Row
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._shared is not None:
            yield self._shared
            self._shared.commit()
            return
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    # -------------------------------------------------------------------- writes

    def save_analysis(
        self,
        analysis_id: str,
        source: str,
        events_received: int,
        events_parsed: int,
        events_suspicious: int,
        threat_detected: bool,
        status: str,
        duration_ms: int = 0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO analyses
                   (analysis_id, source, events_received, events_parsed,
                    events_suspicious, threat_detected, status, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id, source, events_received, events_parsed,
                    events_suspicious, int(threat_detected), status, duration_ms, _now(),
                ),
            )

    def save_incident(self, analysis_id: str, report: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO incidents
                   (incident_id, analysis_id, ip, verdict, severity, confidence,
                    recommended_action, analyzed_by, event_count, suspicious_count,
                    max_event_score, payload, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report["incident_id"], analysis_id, report["ip"], report["verdict"],
                    report["severity"], report["confidence"], report["recommended_action"],
                    report["analyzed_by"], report["event_count"], report["suspicious_count"],
                    report["max_event_score"], json.dumps(report, default=str), _now(),
                ),
            )
            for action in report.get("actions", []):
                connection.execute(
                    """INSERT INTO actions
                       (incident_id, type, target, reason, executed, detail, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report["incident_id"], action["type"], action["target"],
                        action["reason"], int(action["executed"]),
                        action.get("detail", ""), _now(),
                    ),
                )

    def add_to_blocklist(
        self, ip: str, reason: str, mode: str, incident_id: str | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO blocklist
                   (ip, reason, mode, incident_id, created_at) VALUES (?, ?, ?, ?, ?)""",
                (ip, reason, mode, incident_id, _now()),
            )

    # --------------------------------------------------------------------- reads

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def get_analysis_status(self, analysis_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_incidents(
        self, limit: int = 50, ip: str | None = None, analysis_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT incident_id, analysis_id, ip, verdict, severity, "
            "recommended_action, created_at FROM incidents"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if ip:
            clauses.append("ip = ?")
            params.append(ip)
        if analysis_id:
            clauses.append("analysis_id = ?")
            params.append(analysis_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_ip_history(self, ip: str, limit: int = 10) -> dict[str, Any]:
        """Summary of an IP's history. Exposed to the triage agent as a tool."""
        with self.connect() as connection:
            incidents = connection.execute(
                """SELECT incident_id, verdict, severity, recommended_action,
                          suspicious_count, created_at
                   FROM incidents WHERE ip = ? ORDER BY created_at DESC LIMIT ?""",
                (ip, limit),
            ).fetchall()
            blocked = connection.execute(
                "SELECT reason, mode, created_at FROM blocklist WHERE ip = ?", (ip,)
            ).fetchone()
        return {
            "ip": ip,
            "previous_incidents": len(incidents),
            "currently_blocked": blocked is not None,
            "block_detail": dict(blocked) if blocked else None,
            "history": [dict(row) for row in incidents],
        }

    def list_blocklist(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM blocklist ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def is_blocked(self, ip: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM blocklist WHERE ip = ?", (ip,)
            ).fetchone()
        return row is not None
