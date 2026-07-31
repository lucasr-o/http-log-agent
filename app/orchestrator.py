"""Orchestration of the full funnel.

Execution order, and the reason for it:

1. per-event normalization and scoring — deterministic, milliseconds;
2. correlation into dossiers — deterministic, drops whatever holds no suspect;
3. triage agent (LLM) — one call per incident;
4. action agent (LLM) — one call per non-benign incident.

If the LLM is unavailable, has no key configured or the call budget is exhausted,
steps 3 and 4 fall back to a deterministic path. The service never stops
answering because of the model: it degrades the quality of the verdict, not
availability.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from typing import Any, Callable

from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.agents.action import ActionAgent
from app.agents.base import LLMBudget
from app.agents.triage import TriageAgent
from app.config import Settings
from app.db import Database
from app.pipeline.correlator import IncidentDossier, correlate
from app.pipeline.detector import Detector
from app.pipeline.parser import parse_batch
from app.schemas import (
    ActionTaken,
    AnalyzeRequest,
    AnalyzeResponse,
    EventFinding,
    IncidentReport,
    RejectedEvent,
)

logger = logging.getLogger(__name__)

ACTION_ORDER = ["allow", "monitor", "rate_limit", "alert", "block"]

KNOWN_CRAWLERS = (
    "googlebot", "bingbot", "mj12bot", "dotbot", "ahrefsbot", "yandexbot",
    "baiduspider", "duckduckbot", "semrushbot", "mail.ru_bot", "applebot",
)

SEVERITY_ACTION = {"high": "alert", "medium": "alert", "low": "monitor"}


def _most_severe(values: list[str], order: list[str], default: str) -> str:
    ranked = [value for value in values if value in order]
    return max(ranked, key=order.index) if ranked else default


def crawler_claim(dossier: IncidentDossier) -> str:
    """The user agent claiming to be a known crawler, if any.

    This is information for the dossier, never an exemption. The user agent is
    chosen by the client: verifying a crawler requires reverse and forward DNS on
    the IP, which this system does not do. An earlier version cleared the incident
    based on this string, which handed a free pass to `User-Agent: Googlebot` with
    an injection payload attached.
    """
    for event in dossier.events:
        lowered = event.user_agent.lower()
        for crawler in KNOWN_CRAWLERS:
            if crawler in lowered:
                return crawler
    return ""


def deterministic_verdict(
    dossier: IncidentDossier, severity_of: Callable[[float], str] | None = None
) -> dict[str, Any]:
    """Verdict without an LLM, used as the fallback.

    Never closes an incident as benign. The project's criterion is zero false
    negatives: with no ability to investigate, this path classifies as
    `suspicious` and forwards for review, because asserting benignity is the one
    error the system must not make silently.

    The limit of this path is exactly what motivates the agents: the detector says
    "this is improbable", not "this is an attack". Separating unusual legitimate
    traffic from an attack takes context a rule does not have.

    `severity_of` comes from the detector and translates the score into a severity
    using the calibration table. Without it everything lands on `low` — the most
    conservative grade available, since no grade implies clearance.
    """
    score = dossier.max_score
    severity = severity_of(score) if severity_of else "low"
    action = SEVERITY_ACTION[severity]

    # The verdict is never "malicious" on this path. The highest benign score
    # measured (0.711) sits above the attack median (0.585): the distributions
    # overlap, so the score alone does not separate an attack from unusual
    # legitimate traffic. Claiming "malicious" from it would invent confidence the
    # number does not support — and that is precisely the distinction the agents
    # exist to make.
    evidence = next((item.evidence for item in dossier.top_samples() if item.evidence), "")
    claim = crawler_claim(dossier)
    return {
        "verdict": "suspicious",
        "severity": severity,
        "confidence": round(min(0.35 + score / 3, 0.6), 2),
        "attack_types": ["unclassified"],
        "mitre_techniques": [],
        "summary": (
            f"{dossier.suspicious_count} of {dossier.event_count} requests from IP "
            f"{dossier.ip} are improbable under the normal-traffic model "
            f"(max improbability {score:.2f})."
            + (f" Least likely span: {evidence!r}." if evidence else "")
            + (
                f" The user agent claims to be {claim}, which was not verified."
                if claim
                else ""
            )
        ),
        "reasoning": (
            "Deterministic verdict: the language model was unavailable or the "
            "request's call budget was reached. The classification comes only from "
            "the improbability score, which does not distinguish an attack from "
            "unusual legitimate traffic — the distributions overlap. This path "
            "does not issue benign verdicts; review is recommended."
        ),
        "recommended_action": action,
    }


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        detector: Detector,
        database: Database,
        blocker: IPBlocker,
        notifier: Notifier,
        llm_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.database = database
        self.blocker = blocker
        self.notifier = notifier
        self.llm_client = llm_client
        self.triage = TriageAgent(llm_client, database, settings) if llm_client else None
        self.action = (
            ActionAgent(llm_client, database, blocker, notifier, settings)
            if llm_client
            else None
        )

    # ------------------------------------------------------------------ internal

    def _apply_deterministic_action(
        self, dossier: IncidentDossier, verdict: dict[str, Any], dry_run: bool
    ) -> list[dict[str, Any]]:
        """Run the recommended action when no action agent is available."""
        action = verdict.get("recommended_action", "monitor")
        if action in {"allow", "monitor"} or dry_run:
            return []

        performed: list[dict[str, Any]] = []
        if action == "block":
            result = self.blocker.block(dossier.ip, verdict.get("summary", ""))
            if not result.invalid_target:
                self.database.add_to_blocklist(
                    dossier.ip, verdict.get("summary", ""), result.mode, dossier.incident_id
                )
            performed.append(
                {
                    "type": "block", "target": dossier.ip,
                    "reason": verdict.get("summary", ""),
                    "executed": result.executed, "detail": result.detail,
                }
            )
        if action in {"alert", "block"}:
            text = Notifier.format_incident(
                dossier.incident_id, dossier.ip, verdict.get("severity", "medium"),
                verdict.get("verdict", "suspicious"), verdict.get("summary", ""),
                action, list(verdict.get("attack_types", [])),
            )
            notified = self.notifier.send(text)
            performed.append(
                {
                    "type": "alert", "target": dossier.ip,
                    "reason": verdict.get("summary", ""),
                    "executed": notified.sent, "detail": notified.detail,
                }
            )
        return performed

    def _build_report(
        self,
        dossier: IncidentDossier,
        verdict: dict[str, Any],
        actions: list[dict[str, Any]],
        analyzed_by: str,
        final_action: str | None = None,
    ) -> IncidentReport:
        return IncidentReport(
            incident_id=dossier.incident_id,
            ip=dossier.ip,
            window_start=dossier.window_start,
            window_end=dossier.window_end,
            event_count=dossier.event_count,
            suspicious_count=dossier.suspicious_count,
            max_event_score=round(dossier.max_score, 4),
            verdict=verdict.get("verdict", "undetermined"),
            severity=verdict.get("severity", "low"),
            confidence=float(verdict.get("confidence", 0.5)),
            attack_types=list(verdict.get("attack_types", [])),
            mitre_techniques=list(verdict.get("mitre_techniques", [])),
            summary=str(verdict.get("summary", "")),
            reasoning=str(verdict.get("reasoning", "")),
            recommended_action=final_action or verdict.get("recommended_action", "monitor"),
            actions=[ActionTaken(**action) for action in actions],
            analyzed_by=analyzed_by,
            sample_events=[
                EventFinding(
                    ip=item.event.ip,
                    timestamp=item.event.timestamp,
                    method=item.event.method,
                    url=item.event.decoded_url[:500],
                    status=item.event.status,
                    score=round(item.score, 4),
                    user_agent=item.event.user_agent[:200],
                    evidence=item.evidence,
                )
                for item in dossier.top_samples()
            ],
        )

    def _fallback(self, dossier: IncidentDossier, dry_run: bool) -> IncidentReport:
        """Full deterministic verdict, with the recommended action already applied."""
        verdict = deterministic_verdict(dossier, self.detector.severity)
        actions = self._apply_deterministic_action(dossier, verdict, dry_run)
        return self._build_report(dossier, verdict, actions, "deterministic")

    def process_incident(self, dossier: IncidentDossier, budget: LLMBudget, dry_run: bool):
        """Run one incident through both agents, falling back to deterministic mode."""
        if self.triage is None or budget.exhausted:
            return self._fallback(dossier, dry_run)

        # Any failure falls back, not just AgentError. A provider can return 404 for
        # a retired model, 429 for quota, or drop the connection — and a broken
        # provider must degrade the verdict, never the availability of /analyze.
        try:
            verdict = self.triage.analyze(dossier, budget)
        except Exception as exc:  # noqa: BLE001 - deliberate: see comment above
            logger.warning("triage of %s fell back to deterministic mode: %s",
                           dossier.incident_id, exc)
            return self._fallback(dossier, dry_run)

        if verdict.get("verdict") == "benign" or self.action is None or budget.exhausted:
            actions = self._apply_deterministic_action(dossier, verdict, dry_run)
            return self._build_report(dossier, verdict, actions, "llm")

        try:
            plan, executed = self.action.decide(dossier, verdict, budget, dry_run)
            return self._build_report(
                dossier, verdict, executed, "llm", plan.get("final_action")
            )
        except Exception as exc:  # noqa: BLE001 - same reasoning as triage above
            logger.warning("action agent for %s fell back to deterministic mode: %s",
                           dossier.incident_id, exc)
            actions = self._apply_deterministic_action(dossier, verdict, dry_run)
            return self._build_report(dossier, verdict, actions, "llm")

    # -------------------------------------------------------------------- public

    def analyze(
        self, request: AnalyzeRequest, deadline_seconds: float | None = None
    ) -> tuple[AnalyzeResponse, list[IncidentDossier]]:
        """Run the funnel. Returns the response plus the incidents left pending."""
        started = time.monotonic()
        analysis_id = f"anl_{uuid.uuid4().hex[:12]}"
        # `is None` rather than `or`: a deadline of 0 is legitimate and means "do
        # not hold the connection", not "use the default".
        deadline = (
            self.settings.sync_timeout_seconds
            if deadline_seconds is None
            else deadline_seconds
        )

        events, rejected = parse_batch(list(request.events))

        def respond(**fields: Any) -> AnalyzeResponse:
            """Build, persist and return the response — all in one place."""
            response = AnalyzeResponse(
                analysis_id=analysis_id,
                events_received=len(request.events),
                events_parsed=len(events),
                rejected=[RejectedEvent(**item) for item in rejected],
                duration_ms=int((time.monotonic() - started) * 1000),
                **fields,
            )
            self._persist_analysis(response, request.source)
            for report in response.incidents:
                self.database.save_incident(analysis_id, report.model_dump(mode="json"))
            return response

        if not events:
            return respond(
                threat_detected=False,
                message="No event could be normalized.",
            ), []

        scores = self.detector.score_events(events)
        dossiers = correlate(
            events,
            scores,
            self.detector.event_threshold,
            timedelta(minutes=5),
            explain=self.detector.explain,
        )
        near = self.detector.count_near_threshold(scores)

        if not dossiers:
            # No batch is returned as clean without stating how close to the floor
            # the traffic came: under the zero-false-negative policy, "nothing above
            # the cutoff" is a more honest claim than "no threat".
            return respond(
                threat_detected=False,
                events_suspicious=0,
                events_near_threshold=near,
                recommended_action="monitor" if near else "allow",
                message=(
                    f"No event above the escalation floor; {near} landed within "
                    "10% of it."
                    if near
                    else "No event above the escalation floor."
                ),
            ), []

        budget = LLMBudget(self.settings.llm_call_budget)
        reports: list[IncidentReport] = []
        pending: list[IncidentDossier] = []

        for index, dossier in enumerate(dossiers):
            if index > 0 and (time.monotonic() - started) > deadline:
                pending = dossiers[index:]
                break
            reports.append(self.process_incident(dossier, budget, request.dry_run))

        # Incidents left without an agent — call budget exhausted, LLM unavailable
        # or deadline exceeded. None of them was closed as benign, but all received
        # only the weak verdict, and that is declared.
        awaiting = sum(1 for r in reports if r.analyzed_by == "deterministic")
        return respond(
            threat_detected=any(r.verdict != "benign" for r in reports) or bool(pending),
            status="processing" if pending else "completed",
            events_suspicious=sum(d.suspicious_count for d in dossiers),
            events_near_threshold=near,
            incidents=reports,
            incidents_awaiting_agent=awaiting + len(pending),
            recommended_action=_most_severe(
                [r.recommended_action for r in reports], ACTION_ORDER, "monitor"
            ),
            message=(
                f"{len(pending)} incident(s) still processing; "
                f"query GET /incidents?analysis_id={analysis_id}."
                if pending
                else f"{len(reports)} incident(s) analyzed."
            ),
        ), pending

    def process_pending(
        self, analysis_id: str, dossiers: list[IncidentDossier], dry_run: bool
    ) -> None:
        """Process in the background the incidents that did not fit the sync deadline."""
        budget = LLMBudget(self.settings.llm_call_budget)
        for dossier in dossiers:
            try:
                report = self.process_incident(dossier, budget, dry_run)
                self.database.save_incident(analysis_id, report.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - one failure must not abort the queue
                logger.exception("failed to process incident %s", dossier.incident_id)
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE analyses SET status = 'completed' WHERE analysis_id = ?",
                (analysis_id,),
            )

    def _persist_analysis(self, response: AnalyzeResponse, source: str) -> None:
        self.database.save_analysis(
            analysis_id=response.analysis_id,
            source=source,
            events_received=response.events_received,
            events_parsed=response.events_parsed,
            events_suspicious=response.events_suspicious,
            threat_detected=response.threat_detected,
            status=response.status,
            duration_ms=response.duration_ms,
        )
