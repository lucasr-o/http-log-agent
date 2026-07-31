"""API input and output contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["none", "low", "medium", "high", "critical"]
ActionType = Literal["allow", "monitor", "rate_limit", "alert", "block"]


class RawEvent(BaseModel):
    """One log event. Mirrors the columns of the source dataset."""

    ip: str
    time: str | datetime
    url: str
    method: str = "GET"
    protocol: str = "HTTP/1.1"
    status: int = 200
    size: int = 0
    referrer: str = "-"
    user_agent: str = "-"


class AnalyzeRequest(BaseModel):
    events: list[RawEvent | str] = Field(
        ...,
        min_length=1,
        description="Structured events or raw lines in Combined Log Format.",
    )
    source: str = Field(default="unknown", description="Origin of the batch (SIEM, host).")
    dry_run: bool = Field(
        default=False, description="When true, no action is executed."
    )


class RejectedEvent(BaseModel):
    index: int
    reason: str


class EventFinding(BaseModel):
    ip: str
    timestamp: datetime
    method: str
    url: str
    status: int
    score: float = Field(
        ..., description="Normalized improbability; 0.5 is the detection threshold."
    )
    user_agent: str
    evidence: str = Field(
        default="",
        description="Substring of the URL with the highest improbability under the normal-traffic model.",
    )


class ActionTaken(BaseModel):
    type: ActionType
    target: str
    reason: str
    executed: bool
    detail: str = ""


class IncidentReport(BaseModel):
    incident_id: str
    ip: str
    window_start: datetime
    window_end: datetime
    event_count: int
    suspicious_count: int
    max_event_score: float

    verdict: Literal["malicious", "suspicious", "benign", "undetermined"]
    severity: Severity
    confidence: float
    attack_types: list[str] = Field(default_factory=list)
    mitre_techniques: list[str] = Field(default_factory=list)
    summary: str = ""
    reasoning: str = ""

    recommended_action: ActionType = "monitor"
    actions: list[ActionTaken] = Field(default_factory=list)
    analyzed_by: Literal["llm", "deterministic"] = "deterministic"
    sample_events: list[EventFinding] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    analysis_id: str
    threat_detected: bool
    status: Literal["completed", "processing"] = "completed"
    events_received: int
    events_parsed: int
    events_suspicious: int = 0
    events_near_threshold: int = Field(
        default=0,
        description=(
            "Events that landed less than 10% below the escalation floor. They do "
            "not open an incident, but they are reported so that no batch is "
            "returned as clean without that information."
        ),
    )
    rejected: list[RejectedEvent] = Field(default_factory=list)
    incidents: list[IncidentReport] = Field(default_factory=list)
    incidents_awaiting_agent: int = Field(
        default=0,
        description=(
            "Incidents that received only the deterministic verdict — LLM call "
            "budget exhausted, model unavailable or synchronous deadline exceeded. "
            "None of them was closed as benign."
        ),
    )
    recommended_action: ActionType = "allow"
    message: str = ""
    duration_ms: int = 0


class IncidentListItem(BaseModel):
    incident_id: str
    analysis_id: str
    ip: str
    verdict: str
    severity: Severity
    recommended_action: ActionType
    created_at: datetime


class BlocklistEntry(BaseModel):
    ip: str
    reason: str
    mode: str
    created_at: datetime
    incident_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    llm_enabled: bool
    block_mode: str
    detail: dict[str, Any] = Field(default_factory=dict)
