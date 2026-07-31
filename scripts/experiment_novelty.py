"""Experiment: novelty detection on URLs without using labels in training.

Motivation. This project's supervised classifier was trained to reproduce
regex-generated labels, and the error audit showed it does not beat the heuristic:
96% of its false positives are noise and it misses obvious sqlmap payloads. The
task was badly formulated.

Hypothesis. The signal lives in the character structure of the URL — char n-grams
are what took the supervised model to PR-AUC 0.99. If it is there, a character
language model fitted on benign traffic alone should find it without ever seeing a
label or an attack example.

Methodological consequence. The model sees no attack during the fit, so every
attack family is unseen by construction. There is nothing to hold back: detection
is zero-shot for sqli, rce, scanning and bot simultaneously. A regex cannot do
that — it only catches what somebody wrote into it.

Two confounds are measured explicitly:
  1. length — the score is normalized per character, and a baseline using only URL
     length is included for comparison;
  2. split — the fit uses training IPs, the evaluation uses held-out IPs.

Usage:
    python scripts/experiment_novelty.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_model import load_events, split_ips_by_type  # noqa: E402

SEED = 20260728
MAX_ORDER = 4
SMOOTHING = 0.1
BOUNDARY = "\x02"


class CharNGramModel:
    """Character language model with interpolated backoff.

    Stores n-gram counts of order 1..MAX_ORDER and scores a string by its mean
    negative log-likelihood per character. Per-length normalization is
    deliberate: without it every long URL would look anomalous and the model would
    become a disguised length meter.
    """

    def __init__(self, max_order: int = MAX_ORDER, smoothing: float = SMOOTHING) -> None:
        self.max_order = max_order
        self.smoothing = smoothing
        self.context_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(max_order)]
        self.ngram_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(max_order)]
        self.vocabulary: set[str] = set()
        # Decreasing weights by order: long context is more informative when it
        # exists, but needs backoff because most of the time it does not.
        self.weights = np.array([0.4, 0.3, 0.2, 0.1][:max_order])
        self.weights = self.weights / self.weights.sum()

    def fit(self, texts: list[str]) -> "CharNGramModel":
        for text in texts:
            padded = BOUNDARY * (self.max_order - 1) + text
            self.vocabulary.update(text)
            for order in range(1, self.max_order + 1):
                index = order - 1
                for position in range(self.max_order - 1, len(padded)):
                    context = padded[position - index : position]
                    self.context_counts[index][context] += 1
                    self.ngram_counts[index][context + padded[position]] += 1
        return self

    def _probability(self, context: str, char: str, index: int) -> float:
        vocab_size = max(len(self.vocabulary), 1)
        numerator = self.ngram_counts[index].get(context + char, 0) + self.smoothing
        denominator = self.context_counts[index].get(context, 0) + self.smoothing * vocab_size
        return numerator / denominator

    def score(self, text: str) -> float:
        """Mean NLL per character. Higher = less likely under normal traffic."""
        if not text:
            return 0.0
        padded = BOUNDARY * (self.max_order - 1) + text
        total = 0.0
        for position in range(self.max_order - 1, len(padded)):
            mixed = sum(
                weight * self._probability(padded[position - index : position], padded[position], index)
                for index, weight in enumerate(self.weights)
            )
            total += -math.log2(max(mixed, 1e-12))
        return total / len(text)

    def worst_window(self, text: str, width: int = 16) -> tuple[float, str]:
        """Highest-NLL span — locates the substring responsible for the alert.

        This is for explainability: the dossier can quote the exact piece of the URL
        the model found improbable, instead of just a number.
        """
        if len(text) <= width:
            return self.score(text), text
        best_score, best_slice = -1.0, text[:width]
        for start in range(len(text) - width + 1):
            window = text[start : start + width]
            value = self.score(window)
            if value > best_score:
                best_score, best_slice = value, window
        return best_score, best_slice


def report(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    # Recall at 1% false positives: the number that matters for alerting.
    cutoff = float(np.quantile(scores[y_true == 0], 0.99))
    recall_at_1pct = float((scores[y_true == 1] >= cutoff).mean())
    print(
        f"  {name:<34} PR-AUC {pr_auc:.4f}  ROC-AUC {roc_auc:.4f}  "
        f"recall@1%FP {recall_at_1pct:.3f}"
    )
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_1pct_fpr": recall_at_1pct,
        "cutoff_at_1pct_fpr": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/sample.csv.gz"))
    parser.add_argument("--fit-limit", type=int, default=120_000)
    parser.add_argument("--report", type=Path, default=Path("reports/novelty_experiment.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    events, meta = load_events(args.sample)
    train_ips, test_ips = split_ips_by_type(meta, args.seed)
    train_mask = meta["ip"].isin(train_ips).to_numpy()
    test_mask = meta["ip"].isin(test_ips).to_numpy()
    labels = meta["label"].to_numpy()
    types = meta["type"].to_numpy()

    # Fit: benign URLs from training IPs only. No attack and no positive label
    # enters the model.
    fit_urls = [
        event.decoded_url
        for event, keep, label in zip(events, train_mask, labels)
        if keep and label == 0
    ][: args.fit_limit]

    print(f"fitting the character model on {len(fit_urls):,} benign URLs")
    print("(no attack and no positive label enters the fit)\n")
    model = CharNGramModel().fit(fit_urls)
    print(f"vocabulary: {len(model.vocabulary):,} distinct characters")
    print(f"order-{MAX_ORDER} contexts: {len(model.context_counts[-1]):,}\n")

    test_events = [event for event, keep in zip(events, test_mask) if keep]
    y_true = labels[test_mask]
    test_types = types[test_mask]

    print(f"evaluating on {len(test_events):,} held-out events "
          f"({int(y_true.sum())} attacks, {y_true.mean():.2%})\n")

    novelty = np.array([model.score(event.decoded_url) for event in test_events])
    lengths = np.array([len(event.decoded_url) for event in test_events], dtype=float)

    print("RESULTS (no label used to train the novelty model)")
    results = {
        "novelty_char_lm": report("novelty: character LM", y_true, novelty),
        "length_baseline": report("baseline: URL length only", y_true, lengths),
    }

    # Length confound: if the correlation is high, the model is a disguised length
    # meter.
    correlation = float(np.corrcoef(novelty, lengths)[0, 1])
    print(f"\ncorrelation between novelty and length: {correlation:.3f}")
    results["novelty_length_correlation"] = correlation

    print("\nPR-AUC by family (all zero-shot — the model never saw an attack):")
    per_family = {}
    for family in sorted(set(test_types)):
        if family == "benign":
            continue
        mask = (test_types == family) | (y_true == 0)
        subset_true = (test_types[mask] == family).astype(int)
        if subset_true.sum() == 0:
            continue
        pr_auc = float(average_precision_score(subset_true, novelty[mask]))
        cutoff = results["novelty_char_lm"]["cutoff_at_1pct_fpr"]
        recall = float((novelty[mask][subset_true == 1] >= cutoff).mean())
        base = float(subset_true.mean())
        print(
            f"  {family:<10} n={int(subset_true.sum()):>4}  PR-AUC {pr_auc:.4f}  "
            f"recall@1%FP {recall:.3f}  (taxa base {base:.4f})"
        )
        per_family[family] = {
            "support": int(subset_true.sum()), "pr_auc": pr_auc,
            "recall_at_1pct_fpr": recall, "base_rate": base,
        }
    results["per_family"] = per_family

    print("\nHighest-novelty URLs in the held-out set:")
    for index in np.argsort(-novelty)[:12]:
        event = test_events[index]
        _, window = model.worst_window(event.decoded_url)
        print(f"  [{novelty[index]:.2f}] label={test_types[index]:<9} "
              f"{event.decoded_url[:110]}")
        print(f"          least likely span: {window!r}")

    print("\nHighest-novelty benign URLs (source of the false positives):")
    benign_ranked = [i for i in np.argsort(-novelty) if y_true[i] == 0][:6]
    for index in benign_ranked:
        print(f"  [{novelty[index]:.2f}] {test_events[index].decoded_url[:120]}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nreport saved to {args.report}")


if __name__ == "__main__":
    main()
