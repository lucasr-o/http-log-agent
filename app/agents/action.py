"""Agent 2 — Action.

Receives the triage verdict and decides the response. Unlike the triage agent, the
tools here have side effects: `block_ip` writes to the blocklist and `send_alert`
fires the notification. The blast radius is contained by the blocker's `dry_run`
mode, not by the agent being unable to act.

The separation between the two agents is deliberate. Whoever classifies does not
decide, and whoever decides does not reclassify: the action agent receives the
verdict as an input and picks the response proportional to it.
"""

from __future__ import annotations

from typing import Any

from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.agents.base import Agent, LLMBudget, Tool
from app.db import Database
from app.pipeline.correlator import IncidentDossier

SYSTEM_PROMPT = """\
You own incident response for a web server. You receive an incident already \
classified by another analyst and decide which response to apply.

Do not reclassify the incident: the verdict is an input. Your decision is about \
the proportionality of the response.

Response scale, from mildest to most severe:

- `allow` — legitimate traffic. No action. Use for known crawlers and normal \
browsing.
- `monitor` — record without acting. Use when the evidence is weak or the impact \
is nil.
- `rate_limit` — throttle without cutting access. Use for reconnaissance without \
confirmed exploitation, and for aggressive but legitimate bots.
- `alert` — notify the team. Use for medium severity or above, always alongside a \
containment action when one exists.
- `block` — block the IP. Reserve it for confirmed exploitation or a persistent \
attack. Before blocking, check with `check_blocklist` whether the IP is already \
contained, and bear in mind that addresses can be shared behind NAT — blocking a \
mobile carrier IP affects many innocent users.

Check the IP's history before deciding: recidivism justifies escalating. Run the \
tools matching your decision, then call `submit_action_plan` describing what you \
did and why.
"""

ACTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_action": {
            "type": "string",
            "enum": ["allow", "monitor", "rate_limit", "alert", "block"],
            "description": "The most severe action actually applied.",
        },
        "rationale": {
            "type": "string",
            "description": "Why this response is proportional to the incident.",
        },
        "follow_up": {
            "type": "string",
            "description": "Recommendation for the human analyst, if any.",
        },
    },
    "required": ["final_action", "rationale"],
}


def build_tools(
    dossier: IncidentDossier,
    verdict: dict[str, Any],
    database: Database,
    blocker: IPBlocker,
    notifier: Notifier,
    executed: list[dict[str, Any]],
    dry_run: bool,
) -> list[Tool]:
    """Actuation tools. Every execution is recorded in `executed`."""

    def record(action_type: str, target: str, reason: str, ok: bool, detail: str) -> None:
        executed.append(
            {
                "type": action_type,
                "target": target,
                "reason": reason,
                "executed": ok,
                "detail": detail,
            }
        )

    def check_blocklist(arguments: dict[str, Any]) -> Any:
        ip = str(arguments.get("ip") or dossier.ip)
        return {"ip": ip, "blocked": database.is_blocked(ip)}

    def get_ip_history(arguments: dict[str, Any]) -> Any:
        return database.get_ip_history(str(arguments.get("ip") or dossier.ip))

    def block_ip(arguments: dict[str, Any]) -> Any:
        ip = str(arguments.get("ip") or dossier.ip)
        reason = str(arguments.get("reason", "no rationale given"))
        if dry_run:
            record("block", ip, reason, False, "request marked as dry_run")
            return {"applied": False, "detail": "request in dry_run; nothing executed"}
        result = blocker.block(ip, reason)
        if result.invalid_target:
            return {"applied": False, "detail": result.detail}
        database.add_to_blocklist(ip, reason, result.mode, dossier.incident_id)
        record("block", ip, reason, result.executed, result.detail)
        return {"applied": True, "mode": result.mode, "detail": result.detail}

    def rate_limit(arguments: dict[str, Any]) -> Any:
        ip = str(arguments.get("ip") or dossier.ip)
        reason = str(arguments.get("reason", "no rationale given"))
        detail = "rate limit recorded (no proxy integration)"
        record("rate_limit", ip, reason, not dry_run, detail)
        return {"applied": not dry_run, "detail": detail}

    def send_alert(arguments: dict[str, Any]) -> Any:
        message = str(arguments.get("message", "")).strip()
        if not message:
            return {"sent": False, "detail": "the 'message' parameter is required"}
        text = Notifier.format_incident(
            incident_id=dossier.incident_id,
            ip=dossier.ip,
            severity=str(verdict.get("severity", "unknown")),
            verdict=str(verdict.get("verdict", "unknown")),
            summary=message,
            action=str(verdict.get("recommended_action", "monitor")),
            attack_types=list(verdict.get("attack_types", [])),
        )
        if dry_run:
            record("alert", dossier.ip, message, False, "request in dry_run")
            return {"sent": False, "detail": "request in dry_run; nothing sent"}
        result = notifier.send(text)
        record("alert", dossier.ip, message, result.sent, result.detail)
        return {"sent": result.sent, "channel": result.channel, "detail": result.detail}

    def submit_action_plan(arguments: dict[str, Any]) -> Any:  # pragma: no cover
        return arguments

    return [
        Tool(
            name="check_blocklist",
            description="Checks whether the IP is already blocked, to avoid redundant action.",
            input_schema={"type": "object", "properties": {"ip": {"type": "string"}}},
            handler=check_blocklist,
        ),
        Tool(
            name="get_ip_history",
            description="Looks up previous incidents for this IP. Recidivism justifies escalating.",
            input_schema={"type": "object", "properties": {"ip": {"type": "string"}}},
            handler=get_ip_history,
        ),
        Tool(
            name="block_ip",
            description=(
                "Adds the IP to the blocklist. In dry_run mode it only records the "
                "decision. Use it only for confirmed exploitation or persistent attack."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ip": {"type": "string"},
                    "reason": {"type": "string", "description": "Rationale for the block."},
                },
                "required": ["reason"],
            },
            handler=block_ip,
        ),
        Tool(
            name="rate_limit",
            description="Records a rate restriction for the IP, without cutting access.",
            input_schema={
                "type": "object",
                "properties": {
                    "ip": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
            handler=rate_limit,
        ),
        Tool(
            name="send_alert",
            description=(
                "Notifies the security team. Write the message with the evidence "
                "that justifies the alert."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Body of the alert."}
                },
                "required": ["message"],
            },
            handler=send_alert,
        ),
        Tool(
            name="submit_action_plan",
            description="Records the applied response and closes the handling.",
            input_schema=ACTION_PLAN_SCHEMA,
            handler=submit_action_plan,
        ),
    ]


class ActionAgent:
    def __init__(
        self,
        client: Any,
        database: Database,
        blocker: IPBlocker,
        notifier: Notifier,
        settings: Any,
    ) -> None:
        self.client = client
        self.database = database
        self.blocker = blocker
        self.notifier = notifier
        self.settings = settings

    def decide(
        self,
        dossier: IncidentDossier,
        verdict: dict[str, Any],
        budget: LLMBudget,
        dry_run: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        executed: list[dict[str, Any]] = []
        agent = Agent(
            client=self.client,
            model=self.settings.active_model,
            system_prompt=SYSTEM_PROMPT,
            tools=build_tools(
                dossier, verdict, self.database, self.blocker,
                self.notifier, executed, dry_run,
            ),
            terminal_tool="submit_action_plan",
            max_tokens=self.settings.llm_max_tokens,
            effort=self.settings.llm_effort,
            max_iterations=self.settings.llm_max_tool_iterations,
        )
        message = (
            "Decide the response for the incident below.\n\n"
            f"TRIAGE VERDICT\n"
            f"  classification: {verdict.get('verdict')}\n"
            f"  severity: {verdict.get('severity')}\n"
            f"  confidence: {verdict.get('confidence')}\n"
            f"  types: {', '.join(verdict.get('attack_types', [])) or 'unclassified'}\n"
            f"  suggested action: {verdict.get('recommended_action')}\n"
            f"  summary: {verdict.get('summary')}\n"
            f"  rationale: {verdict.get('reasoning')}\n\n"
            f"{dossier.to_prompt()}"
        )
        result = agent.run(message, budget)
        output = result.output
        output["_meta"] = {
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "llm_calls": result.llm_calls,
        }
        return output, executed
