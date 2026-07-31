"""Agent 1 — Triage.

Receives an incident dossier and decides what it is. It does not receive the raw
event list: if it wants more data it has to ask through a tool. That ability to
investigate on demand is what separates an agent from a prompt.
"""

from __future__ import annotations

import re
from typing import Any

from app.agents.base import Agent, LLMBudget, Tool
from app.db import Database
from app.pipeline.correlator import IncidentDossier

SYSTEM_PROMPT = """\
You are a senior security analyst triaging incidents in the access logs of an \
e-commerce web server.

You receive an incident dossier: an IP, a time window, aggregate statistics and \
the highest-scoring events according to a classifier. That classifier is a \
high-recall, moderate-precision filter — it makes mistakes, and part of your job \
is to correct it in both directions.

The available tools let you dig deeper before deciding: pull more events from the \
window, search the URLs by pattern, aggregate by field and look up the IP's \
history. Use them when the dossier is ambiguous. Do not use them when the \
evidence is already conclusive — every call costs time.

Classification criteria:

- SQLi, RCE, directory traversal, XSS: malicious. High or critical severity \
depending on evidence of success (status 200 with a large body is worse than 404).
- Directory scanning and brute-forcing of well-known paths (/wp-login.php, \
/.env, /.git): malicious, but normally medium severity — it is reconnaissance, \
not compromise.
- Legitimate crawlers (Googlebot, Bingbot, MJ12bot, DotBot, AhrefsBot, \
Mail.RU_Bot) requesting /robots.txt or browsing the catalog are NOT an attack: \
classify as benign and recommend `allow`. The automatic labeler that generated \
the training data marked that traffic as hostile; it was wrong, and blocking a \
legitimate crawler causes real business damage.
- But the user agent is chosen by whoever makes the request and the system does \
not verify reverse DNS. A claim of being a crawler only holds if the window's \
BEHAVIOR is consistent with it: catalog paths and robots.txt, no injection \
payload, no probing of administrative paths. A Googlebot user agent accompanied \
by `UNION SELECT` or `/wp-admin` is spoofing, and there the verdict follows the \
behavior, never the string. State in `reasoning` which of the two cases applies.
- Normal browsing traffic, even in high volume, is benign.

Always finish by calling `submit_verdict`. Write `summary` and `reasoning` so an \
on-call analyst understands in ten seconds what happened and why. Quote concrete \
evidence from the dossier — URL, status, user agent — instead of generic claims.
"""

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["malicious", "suspicious", "benign"],
            "description": "Final classification of the incident.",
        },
        "severity": {
            "type": "string",
            "enum": ["none", "low", "medium", "high", "critical"],
            "description": "Potential impact. Use 'none' for benign traffic.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the classification, from 0 to 1.",
        },
        "attack_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "E.g. sqli, rce, path_traversal, scanning, crawler.",
        },
        "mitre_techniques": {
            "type": "array",
            "items": {"type": "string"},
            "description": "MITRE ATT&CK technique IDs, e.g. T1190.",
        },
        "summary": {
            "type": "string",
            "description": "One or two sentences on what happened.",
        },
        "reasoning": {
            "type": "string",
            "description": "Rationale quoting the concrete evidence used.",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["allow", "monitor", "rate_limit", "alert", "block"],
            "description": "Response proportional to the verdict.",
        },
    },
    "required": [
        "verdict", "severity", "confidence", "attack_types",
        "summary", "reasoning", "recommended_action",
    ],
}


def build_tools(dossier: IncidentDossier, database: Database) -> list[Tool]:
    """Read-only tools, always scoped to the incident at hand."""

    def get_sample_events(arguments: dict[str, Any]) -> Any:
        limit = min(int(arguments.get("limit", 10)), 50)
        only_suspicious = bool(arguments.get("only_suspicious", True))
        source = (
            [item.event for item in dossier.top_samples(limit)]
            if only_suspicious
            else dossier.events[:limit]
        )
        return {
            "count": len(source),
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "method": event.method,
                    "url": event.decoded_url[:500],
                    "status": event.status,
                    "size": event.size,
                    "user_agent": event.user_agent[:200],
                    "referrer": event.referrer[:200],
                }
                for event in source
            ],
        }

    def search_events(arguments: dict[str, Any]) -> Any:
        pattern = str(arguments.get("pattern", ""))
        if not pattern:
            return {"error": "the 'pattern' parameter is required"}
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return {"error": f"invalid regular expression: {exc}"}
        matches = [
            {
                "timestamp": event.timestamp.isoformat(),
                "url": event.decoded_url[:400],
                "status": event.status,
            }
            for event in dossier.events
            if regex.search(event.decoded_url) or regex.search(event.user_agent)
        ]
        return {"pattern": pattern, "match_count": len(matches), "matches": matches[:25]}

    def count_by_field(arguments: dict[str, Any]) -> Any:
        field = str(arguments.get("field", "status"))
        extractors = {
            "status": lambda event: str(event.status),
            "method": lambda event: event.method,
            "path": lambda event: event.path,
            "user_agent": lambda event: event.user_agent[:120],
        }
        if field not in extractors:
            return {"error": f"invalid field; use one of {sorted(extractors)}"}
        counts: dict[str, int] = {}
        for event in dossier.events:
            key = extractors[field](event)
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
        return {"field": field, "distinct": len(counts), "counts": dict(top)}

    def get_ip_history(arguments: dict[str, Any]) -> Any:
        ip = str(arguments.get("ip") or dossier.ip)
        return database.get_ip_history(ip)

    def submit_verdict(arguments: dict[str, Any]) -> Any:  # pragma: no cover - terminal
        return arguments

    return [
        Tool(
            name="get_sample_events",
            description=(
                "Returns raw events from this window. Use it to inspect payloads "
                "the dossier summary truncated."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum events (up to 50)."},
                    "only_suspicious": {
                        "type": "boolean",
                        "description": "When false, also includes unflagged traffic.",
                    },
                },
            },
            handler=get_sample_events,
        ),
        Tool(
            name="search_events",
            description=(
                "Searches a regular expression across the window's URLs and user "
                "agents. Use it to confirm or rule out a specific attack family."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."}
                },
                "required": ["pattern"],
            },
            handler=search_events,
        ),
        Tool(
            name="count_by_field",
            description=(
                "Aggregates the window's events by status, method, path or "
                "user_agent. Use it to tell scanning apart from browsing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["status", "method", "path", "user_agent"],
                    }
                },
                "required": ["field"],
            },
            handler=count_by_field,
        ),
        Tool(
            name="get_ip_history",
            description=(
                "Looks up previous incidents for this IP and whether it is already "
                "on the blocklist. Recidivism raises severity."
            ),
            input_schema={
                "type": "object",
                "properties": {"ip": {"type": "string"}},
            },
            handler=get_ip_history,
        ),
        Tool(
            name="submit_verdict",
            description="Records the final classification and ends triage.",
            input_schema=VERDICT_SCHEMA,
            handler=submit_verdict,
        ),
    ]


class TriageAgent:
    def __init__(self, client: Any, database: Database, settings: Any) -> None:
        self.client = client
        self.database = database
        self.settings = settings

    def analyze(self, dossier: IncidentDossier, budget: LLMBudget) -> dict[str, Any]:
        agent = Agent(
            client=self.client,
            model=self.settings.active_model,
            system_prompt=SYSTEM_PROMPT,
            tools=build_tools(dossier, self.database),
            terminal_tool="submit_verdict",
            max_tokens=self.settings.llm_max_tokens,
            effort=self.settings.llm_effort,
            max_iterations=self.settings.llm_max_tool_iterations,
        )
        result = agent.run(
            "Triage the incident below.\n\n" + dossier.to_prompt(), budget
        )
        output = result.output
        output["_meta"] = {
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "llm_calls": result.llm_calls,
        }
        return output
