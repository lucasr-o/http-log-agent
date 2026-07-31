"""Derive a Git-versionable sample from labeled.csv (2.8 GB).

The original dataset holds 9,281,184 rows and 0.032% attacks. Uniform sampling
would lose almost every positive, so we keep 100% of the attack events and
subsample the benign ones.

The detector scores one request at a time, so dropping benign rows distorts
nothing: what matters is the diversity of normal URLs, and 300 thousand of them
cover the site's vocabulary comfortably.

Usage:
    python scripts/sample_dataset.py --source labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "ip",
    "time",
    "method",
    "url",
    "protocol",
    "status",
    "size",
    "referrer",
    "user_agent",
    "extra",
    "no",
    "label",
    "type",
]

CHUNK_SIZE = 500_000
SEED = 20260728


def sample(source: Path, out: Path, benign_target: int, seed: int) -> pd.DataFrame:
    if not source.exists():
        sys.exit(f"dataset not found: {source}")

    # First pass: count benign rows to derive the sampling rate.
    benign_total = 0
    attack_total = 0
    for chunk in pd.read_csv(
        source, chunksize=CHUNK_SIZE, usecols=["type"], dtype=str, on_bad_lines="skip"
    ):
        is_benign = chunk["type"] == "benign"
        benign_total += int(is_benign.sum())
        attack_total += int((~is_benign).sum())

    if benign_total == 0:
        sys.exit("no benign row found — check the source file")

    rate = min(1.0, benign_target / benign_total)
    print(f"benign={benign_total:,}  attacks={attack_total:,}  rate={rate:.5f}")

    # Second pass: keep every attack plus a sample of the benign rows.
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source, chunksize=CHUNK_SIZE, dtype=str, on_bad_lines="skip"
    ):
        chunk = chunk.reindex(columns=COLUMNS)
        is_benign = chunk["type"] == "benign"
        keep = ~is_benign | (rng.random(len(chunk)) < rate)
        parts.append(chunk[keep])

    frame = pd.concat(parts, ignore_index=True)
    frame["no"] = pd.to_numeric(frame["no"], errors="coerce")
    frame = frame.dropna(subset=["no", "ip", "url", "type"])
    frame["no"] = frame["no"].astype("int64")
    frame = frame.sort_values("no").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, compression="gzip")

    counts = frame["type"].value_counts()
    print(f"\nsample written to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"rows: {len(frame):,}")
    for label, count in counts.items():
        print(f"  {label:<10} {count:>8,}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("labeled.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/sample.csv.gz"))
    parser.add_argument("--benign-target", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    sample(args.source, args.out, args.benign_target, args.seed)


if __name__ == "__main__":
    main()
