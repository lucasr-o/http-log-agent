"""Send real dataset events to the API and print the result.

Doubles as a demonstration and as an end-to-end smoke check. The events come from
`data/sample.csv.gz`, meaning this is real traffic from the original dataset and
not a synthetic payload.

Usage:
    python scripts/demo.py --scenario mixed
    python scripts/demo.py --scenario sqli --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

FIELDS = ["ip", "time", "method", "url", "protocol", "status", "size", "referrer", "user_agent"]


def load_events(sample: Path, scenario: str, limit: int) -> list[dict]:
    frame = pd.read_csv(sample, dtype=str, keep_default_na=False)

    if scenario == "benign":
        chosen = frame[frame["type"] == "benign"].head(limit)
    elif scenario == "mixed":
        # A handful of attacks drowned in normal traffic — the realistic case.
        attacks = frame[frame["type"].isin(["sqli", "rce", "scanning"])].head(8)
        benign = frame[frame["type"] == "benign"].head(max(limit - len(attacks), 0))
        chosen = pd.concat([benign, attacks]).sort_values("no")
    else:
        chosen = frame[frame["type"] == scenario].head(limit)

    if chosen.empty:
        sys.exit(f"no event of type {scenario!r} found in {sample}")

    events = []
    for record in chosen[FIELDS].to_dict("records"):
        record["status"] = int(record["status"] or 0)
        record["size"] = int(float(record["size"] or 0))
        events.append(record)
    return events


LABEL_WIDTH = 15


def field(label: str, value: object) -> None:
    print(f"{label + ':':<{LABEL_WIDTH}}{value}")


def summarize(body: dict) -> None:
    field("analysis", f"{body['analysis_id']} ({body['status']})")
    field("events", f"{body['events_parsed']} normalized of {body['events_received']}")
    field("suspicious", body["events_suspicious"])
    field("near floor", body["events_near_threshold"])
    field("no agent", body["incidents_awaiting_agent"])
    field("threat", body["threat_detected"])
    field("global action", body["recommended_action"])
    field("duration", f"{body['duration_ms']} ms")

    if not body["incidents"]:
        print("\nno incident opened.")
        return

    for incident in body["incidents"]:
        print("\n" + "-" * 72)
        field("incident", incident["incident_id"])
        field("source", incident["ip"])
        field("window", f"{incident['window_start']} .. {incident['window_end']}")
        field(
            "events",
            f"{incident['suspicious_count']} suspicious of {incident['event_count']} "
            f"(max score {incident['max_event_score']})",
        )
        field("analyzed by", incident["analyzed_by"])
        field(
            "verdict",
            f"{incident['verdict']} / {incident['severity']} "
            f"(confidence {incident['confidence']})",
        )
        if incident["attack_types"]:
            field("types", ", ".join(incident["attack_types"]))
        if incident["mitre_techniques"]:
            field("MITRE", ", ".join(incident["mitre_techniques"]))
        field("action", incident["recommended_action"])
        print()
        field("summary", incident["summary"])
        if incident["reasoning"]:
            field("rationale", incident["reasoning"])
        for sample in incident["sample_events"][:3]:
            print(f"  [{sample['score']:.3f}] {sample['status']} {sample['url'][:96]}")
            if sample.get("evidence"):
                print(f"          least likely span: {sample['evidence']!r}")
        for action in incident["actions"]:
            mark = "executed" if action["executed"] else "recorded"
            print(f"  action {mark}: {action['type']} -> {action['target']} ({action['detail']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dev-key-change-me")
    parser.add_argument("--sample", type=Path, default=Path("data/sample.csv.gz"))
    parser.add_argument(
        "--scenario", default="mixed",
        choices=["benign", "mixed", "sqli", "rce", "scanning", "bot"],
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="imprime a resposta crua")
    args = parser.parse_args()

    events = load_events(args.sample, args.scenario, args.limit)
    # With --json, stdout carries only the JSON so it can be piped.
    print(
        f"sending {len(events)} events (scenario: {args.scenario}) to {args.url}\n",
        file=sys.stderr if args.json else sys.stdout,
    )

    try:
        response = httpx.post(
            f"{args.url}/analyze",
            json={"events": events, "source": f"demo:{args.scenario}"},
            headers={"X-API-Key": args.api_key},
            timeout=180.0,
        )
    except httpx.HTTPError as exc:
        sys.exit(f"falha ao contatar a API: {exc}")

    if response.status_code not in (200, 202):
        sys.exit(f"HTTP {response.status_code}: {response.text[:500]}")

    body = response.json()
    if args.json:
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        summarize(body)


if __name__ == "__main__":
    main()
