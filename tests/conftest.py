"""Shared fixtures.

No test touches the network. The Anthropic client is replaced by a double that
returns scripted responses, which makes it possible to exercise the tool loop —
including its error paths — deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.config import Settings
from app.db import Database
from app.pipeline.correlator import correlate
from app.pipeline.parser import parse_event

BASE_TIME = datetime(2019, 1, 23, 3, 0, 0, tzinfo=timezone.utc)

SQLI_URL = (
    "/image/29000?name=6aba3c.jpg&wh=200x200' AND 3953=(SELECT UPPER(XMLType("
    "CHR(60)||CHR(58)||CHR(113))) FROM DUAL)--"
)


# -------------------------------------------------------------------- doubles


@dataclass
class FakeBlock:
    type: str
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "tool_use"


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        # `messages` is the same list the loop keeps mutating; storing the
        # reference would make every recorded call look like the final state.
        snapshot = dict(kwargs)
        snapshot["messages"] = [dict(message) for message in kwargs.get("messages", [])]
        self.calls.append(snapshot)
        if not self._responses:
            raise AssertionError("the double ran out of scripted responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.messages = FakeMessages(responses)


def tool_use(name: str, arguments: dict[str, Any], block_id: str = "tu_1") -> FakeResponse:
    return FakeResponse(
        content=[FakeBlock(type="tool_use", id=block_id, name=name, input=arguments)]
    )


def text_response(text: str, stop_reason: str = "end_turn") -> FakeResponse:
    return FakeResponse(
        content=[FakeBlock(type="text", text=text)], stop_reason=stop_reason
    )


VERDICT_MALICIOUS = {
    "verdict": "malicious",
    "severity": "critical",
    "confidence": 0.95,
    "attack_types": ["sqli"],
    "mitre_techniques": ["T1190"],
    "summary": "SQL injection confirmed through the name parameter.",
    "reasoning": "The payload holds UPPER(XMLType(...)) FROM DUAL, typical of sqlmap.",
    "recommended_action": "block",
}

VERDICT_BENIGN = {
    "verdict": "benign",
    "severity": "none",
    "confidence": 0.9,
    "attack_types": ["crawler"],
    "mitre_techniques": [],
    "summary": "Legitimate crawler requesting robots.txt.",
    "reasoning": "The user agent matches MJ12bot and the only path is /robots.txt.",
    "recommended_action": "allow",
}


class StubDetector:
    """Rule-based scoring detector, so tests do not depend on the trained artifact.

    Reproduces the novelty detector's interface: a normalized score where 0.5 is the
    threshold, plus localization of the responsible span.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.event_threshold = threshold
        self.is_loaded = True
        self.load_error = None
        self.applied_fpr = 0.002
        self.metadata: dict[str, Any] = {"approach": "stub", "variant": "url_only"}

    def score_events(self, events):
        scores = []
        for event in events:
            url = event.decoded_url.lower()
            if any(token in url for token in ("select", "union", "shell_exec", "../")):
                scores.append(0.97)
            elif any(token in url for token in ("wp-login", "/.env", "/.git")):
                scores.append(0.80)
            elif "nearmiss" in url:
                scores.append(0.47)  # near-miss band, below the floor
            else:
                scores.append(0.02)
        return scores

    def explain(self, event):
        url = event.decoded_url
        for token in ("UNION ALL SELECT", "SELECT", "shell_exec", "../"):
            position = url.upper().find(token.upper())
            if position >= 0:
                return url[position : position + 16]
        return url[:16]

    def severity(self, score: float) -> str:
        if score >= 0.9:
            return "high"
        if score >= 0.7:
            return "medium"
        return "low"

    def count_near_threshold(self, scores) -> int:
        return sum(1 for score in scores if 0.45 <= score < self.event_threshold)


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        api_key="test-key",
        database_path=tmp_path / "test.db",
        model_path=tmp_path / "missing.joblib",
        llm_call_budget=10,
        llm_max_tool_iterations=5,
        block_mode="dry_run",
        anthropic_api_key="",
    )


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


def make_event(
    url: str = "/product/1",
    ip: str = "10.0.0.1",
    status: int = 200,
    offset: int = 0,
    user_agent: str = "Mozilla/5.0",
    method: str = "GET",
    size: int = 1024,
):
    return parse_event(
        {
            "ip": ip,
            "time": (BASE_TIME + timedelta(seconds=offset)).isoformat(),
            "method": method,
            "url": url,
            "protocol": "HTTP/1.1",
            "status": status,
            "size": size,
            "referrer": "-",
            "user_agent": user_agent,
        }
    )


@pytest.fixture
def benign_events():
    return [make_event(url=f"/product/{i}", offset=i * 10) for i in range(6)]


@pytest.fixture
def attack_events():
    events = [make_event(url=f"/product/{i}", ip="203.0.113.9", offset=i * 5) for i in range(4)]
    events.append(make_event(url=SQLI_URL, ip="203.0.113.9", status=200, offset=30))
    events.append(
        make_event(url="/index.php?cmd=shell_exec", ip="203.0.113.9", status=400, offset=40)
    )
    return events


@pytest.fixture
def attack_dossier(attack_events):
    detector = StubDetector()
    scores = detector.score_events(attack_events)
    dossiers = correlate(
        attack_events, scores, detector.event_threshold, explain=detector.explain
    )
    assert dossiers, "expected at least one incident"
    return dossiers[0]
