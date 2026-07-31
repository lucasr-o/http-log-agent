"""Audit the novelty detector's errors on the held-out set.

Two questions, both needed to know whether the detector is trustworthy:

1. Are the false positives unusual legitimate traffic, or attacks the regex
   labeler let through? An FP that trips an independent attack signature is
   evidence of a wrong label, not of a wrong detector.
2. What escapes? The false negatives show where the character-improbability
   approach has a structural limit.

This script was originally written to audit a supervised classifier. The audit
showed that 96% of its false positives were noise and that it missed obvious
sqlmap payloads — which is what motivated switching to novelty detection. It
remains as a continuous verification tool.

Usage:
    python scripts/inspect_errors.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_model import load_events, split_ips_by_type  # noqa: E402

# Conservative signatures, used only to audit — never to train.
AUDIT_SIGNATURES = {
    "sql": re.compile(
        r"(\bunion\b.{0,20}\bselect\b|\bselect\b.{0,40}\bfrom\b|information_schema|"
        r"\bor\b\s+\d+\s*=\s*\d+|sleep\s*\(|benchmark\s*\(|waitfor\s+delay|"
        r"\bdrop\s+table\b|xmltype|utl_inaddr|dbms_pipe|pg_sleep|\border\s+by\b\s*\d)",
        re.IGNORECASE,
    ),
    "rce_cmd": re.compile(
        r"(shell_exec|call_user_func|system\s*\(|passthru|/bin/(sh|bash)|"
        r"wget\s+http|curl\s+http|chmod\s+777|\bexec\s*\()",
        re.IGNORECASE,
    ),
    "traversal": re.compile(r"(\.\./|\.\.%2f|%2e%2e|/etc/passwd|boot\.ini)", re.IGNORECASE),
    "xss": re.compile(r"(<script|javascript:|onerror\s*=|onload\s*=|alert\s*\()", re.IGNORECASE),
    "sensitive_path": re.compile(
        r"(/\.git|/\.env|/\.svn|/\.idea|/\.aws|/wp-admin|/wp-login|/phpmyadmin|"
        r"/xmlrpc\.php|/config\.php|/backup|\.sql$|\.bak$)",
        re.IGNORECASE,
    ),
    "scanner_ua": re.compile(
        r"(sqlmap|nikto|nmap|masscan|dirbuster|gobuster|acunetix|nessus|zgrab)",
        re.IGNORECASE,
    ),
}


def audit(url: str, user_agent: str) -> list[str]:
    return [
        name
        for name, pattern in AUDIT_SIGNATURES.items()
        if pattern.search(user_agent if name == "scanner_ua" else url)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/sample.csv.gz"))
    parser.add_argument("--model", type=Path, default=Path("models/detector.joblib"))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    detector = joblib.load(args.model)["detector"]
    events, meta = load_events(args.sample)
    _, test_ips = split_ips_by_type(meta, args.seed)
    test_mask = meta["ip"].isin(test_ips).to_numpy()

    test_events = [event for event, keep in zip(events, test_mask) if keep]
    y_true = meta["label"].to_numpy()[test_mask]
    types = meta["type"].to_numpy()[test_mask]

    scores = np.array([detector.normalized_score(e.decoded_url) for e in test_events])
    flagged = scores >= 0.5

    fp_idx = np.where(flagged & (y_true == 0))[0]
    fn_idx = np.where(~flagged & (y_true == 1))[0]

    print(f"test set: {len(test_events):,} events, {int(y_true.sum())} attacks")
    print(f"false positives: {len(fp_idx)}   false negatives: {len(fn_idx)}\n")

    audited = [(i, audit(test_events[i].decoded_url, test_events[i].user_agent)) for i in fp_idx]
    with_signature = [(i, hits) for i, hits in audited if hits]
    without = [i for i, hits in audited if not hits]

    print("=" * 78)
    print("FALSE POSITIVES")
    print("=" * 78)
    share = len(with_signature) / len(fp_idx) if len(fp_idx) else 0.0
    print(
        f"{len(with_signature)} of {len(fp_idx)} ({share:.1%}) trip an independent "
        "attack signature"
    )
    print("=> labeled 'benign' by the dataset, but most likely attacks\n")
    for name, count in Counter(n for _, hits in with_signature for n in hits).most_common():
        print(f"  {name:<18} {count:>4}")

    if with_signature:
        print("\ndataset label most likely wrong:")
        for i, hits in with_signature[: args.show]:
            event = test_events[i]
            evidence = detector.explain(event.decoded_url)
            print(f"  [{scores[i]:.3f}] {','.join(hits)}")
            print(f"          {event.method} {event.status} {event.decoded_url[:130]}")
            print(f"          least likely span: {evidence!r}")

    print("\nunusual legitimate traffic (the triage agent should dismiss it):")
    for i in sorted(without, key=lambda j: -scores[j])[: args.show]:
        event = test_events[i]
        evidence = detector.explain(event.decoded_url)
        print(f"  [{scores[i]:.3f}] {event.status} {event.decoded_url[:130]}")
        print(f"          least likely span: {evidence!r}")

    print("\n" + "=" * 78)
    print("FALSE NEGATIVES (structural limit of the approach)")
    print("=" * 78)
    print(Counter(types[fn_idx]).most_common())
    for i in sorted(fn_idx, key=lambda j: -scores[j])[: args.show]:
        event = test_events[i]
        print(f"  [{scores[i]:.3f}] type={types[i]:<9} {event.status} "
              f"{event.decoded_url[:130]}")


if __name__ == "__main__":
    main()
