"""Correlation of suspicious events into incident dossiers.

This is the piece that makes the LLM cost viable. A batch of 10,000 events
typically yields a few dozen suspects, which group into a handful of incidents.
The agent receives the dossier — aggregate statistics plus a handful of
representative events — and not the raw list. The difference between sending 400
log lines and sending 40 lines of summary is the difference between the project
being runnable and not.

A dossier carries the IP's whole window, not just the suspicious events: knowing
that an IP made 3 malicious requests among 400 legitimate ones is different
information from knowing it made 3 out of 3.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

from app.pipeline.parser import LogEvent

MAX_SAMPLE_EVENTS = 5
MAX_URL_CHARS = 300
DEFAULT_WINDOW = timedelta(minutes=5)


def group_by_ip_window(
    events: Sequence[LogEvent], window: timedelta = DEFAULT_WINDOW
) -> dict[tuple[str, int], list[int]]:
    """Group event positions by (IP, time bucket).

    The bucket derives from the absolute timestamp rather than from the batch's
    first event, so the same event always lands in the same group no matter how
    the client sliced the request. Without that, the outcome of an analysis would
    depend on the size of the batch submitted.

    Positions are returned instead of events so callers can index parallel lists —
    scores, labels — without needing a map keyed on object identity.
    """
    seconds = max(int(window.total_seconds()), 1)
    groups: dict[tuple[str, int], list[int]] = {}
    for position, event in enumerate(events):
        bucket = int(event.timestamp.timestamp()) // seconds
        groups.setdefault((event.ip, bucket), []).append(position)
    return groups


@dataclass(slots=True)
class ScoredEvent:
    event: LogEvent
    score: float
    evidence: str = ""


@dataclass(slots=True)
class IncidentDossier:
    """Summary of one (IP, window) pair — the agents' unit of work."""

    incident_id: str
    ip: str
    window_start: datetime
    window_end: datetime
    events: list[LogEvent]
    suspicious: list[ScoredEvent]
    stats: dict = field(default_factory=dict)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def suspicious_count(self) -> int:
        return len(self.suspicious)

    @property
    def max_score(self) -> float:
        return max((item.score for item in self.suspicious), default=0.0)

    def top_samples(self, limit: int = MAX_SAMPLE_EVENTS) -> list[ScoredEvent]:
        return sorted(self.suspicious, key=lambda item: item.score, reverse=True)[:limit]

    def to_prompt(self) -> str:
        """Textual rendering of the dossier, which is what the agent actually reads."""
        lines = [
            f"INCIDENT {self.incident_id}",
            f"Source IP: {self.ip}",
            f"Window: {self.window_start.isoformat()} .. {self.window_end.isoformat()}",
            f"Requests in window: {self.event_count}",
            f"Flagged as improbable: {self.suspicious_count}"
            f" (max improbability {self.max_score:.3f}; 0.50 is the threshold)",
            "",
            "The detector is a character model fitted on normal traffic only.",
            "A high score means the request does not look like this site's usual",
            "traffic — which includes attacks, but also includes unusual legitimate",
            "traffic. Telling the two apart is your job.",
        ]

        stats = self.stats
        lines.append("")
        lines.append("WINDOW STATISTICS")
        lines.append(f"  HTTP status: {stats.get('status_counts', {})}")
        lines.append(f"  methods: {stats.get('method_counts', {})}")
        lines.append(f"  distinct paths: {stats.get('unique_paths', 0)}")
        lines.append(f"  error rate: {stats.get('error_rate', 0):.2f}")
        lines.append(f"  user agents: {stats.get('user_agents', [])}")
        if stats.get("top_paths"):
            lines.append("  most frequent paths:")
            for path, count in stats["top_paths"]:
                lines.append(f"    {count:>4}x  {path[:120]}")

        lines.append("")
        lines.append(f"LEAST LIKELY EVENTS (up to {MAX_SAMPLE_EVENTS})")
        for item in self.top_samples():
            event = item.event
            url = event.decoded_url[:MAX_URL_CHARS]
            truncated = "…" if len(event.decoded_url) > MAX_URL_CHARS else ""
            lines.append(
                f"  [{item.score:.3f}] {event.timestamp.isoformat()} "
                f"{event.method} {event.status} {url}{truncated}"
            )
            lines.append(f"          user agent: {event.user_agent[:160]}")
            if item.evidence:
                # The span of highest improbability, located by the model. This is
                # the concrete evidence the agent can quote in its rationale.
                lines.append(f"          least likely span: {item.evidence!r}")
        return "\n".join(lines)


def _window_stats(events: Sequence[LogEvent]) -> dict:
    statuses = Counter(event.status for event in events)
    methods = Counter(event.method for event in events)
    paths = Counter(event.path for event in events)
    agents = sorted({event.user_agent for event in events})
    errors = sum(1 for event in events if event.status >= 400)
    return {
        "status_counts": dict(statuses.most_common(6)),
        "method_counts": dict(methods),
        "unique_paths": len(paths),
        "error_rate": errors / len(events) if events else 0.0,
        "user_agents": [agent[:120] for agent in agents[:4]],
        "top_paths": paths.most_common(5),
    }


def correlate(
    events: Sequence[LogEvent],
    scores: Sequence[float],
    threshold: float,
    window: timedelta = DEFAULT_WINDOW,
    explain: Callable[[LogEvent], str] | None = None,
) -> list[IncidentDossier]:
    """Group events into incidents, keeping only the groups that hold suspects.

    A group with no event above the threshold does not become an incident and
    consumes no LLM call — that is how the benign case stays cheap.

    This is the only point in the system where traffic stops being examined, which
    is why the floor is chosen by incident recall rather than by precision: see
    `app.pipeline.detector`. Callers also receive the count of events that landed
    just below the cutoff, so no batch is returned as clean without that fact.

    `explain` is optional and only ever receives the events that will be attached
    to the dossier, never the whole batch: locating the improbable substring costs
    more than scoring, and only pays off for what the agent will actually read. It
    arrives as a callable so the correlator does not depend on the detector.
    """
    if len(events) != len(scores):
        raise ValueError("events and scores must have the same length")

    seconds = int(window.total_seconds())
    dossiers: list[IncidentDossier] = []

    for (ip, bucket), positions in sorted(
        group_by_ip_window(events, window).items(), key=lambda item: item[0]
    ):
        group = [events[position] for position in positions]
        suspicious = [
            ScoredEvent(event=events[position], score=scores[position])
            for position in positions
            if scores[position] >= threshold
        ]
        if not suspicious:
            continue
        if explain is not None:
            for item in sorted(suspicious, key=lambda i: i.score, reverse=True)[
                :MAX_SAMPLE_EVENTS
            ]:
                item.evidence = explain(item.event)
        start = datetime.fromtimestamp(bucket * seconds, tz=group[0].timestamp.tzinfo)
        dossiers.append(
            IncidentDossier(
                incident_id=f"inc_{uuid.uuid4().hex[:12]}",
                ip=ip,
                window_start=start,
                window_end=start + window,
                events=sorted(group, key=lambda event: event.timestamp),
                suspicious=suspicious,
                stats=_window_stats(group),
            )
        )

    dossiers.sort(key=lambda dossier: dossier.max_score, reverse=True)
    return dossiers
