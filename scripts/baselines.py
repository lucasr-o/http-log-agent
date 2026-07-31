"""Comparison baselines. Not part of the service.

They exist to answer, with numbers, the question that decided this project's
architecture: does a supervised classifier trained on this dataset's labels add
anything over the regex heuristic that generated those same labels?

The measured answer is no. Both stay here as evidence for why detection was moved
to an unsupervised approach, and both are re-evaluated on every training run so
the claim stays verifiable.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.pipeline.parser import LogEvent

SEED = 20260728
SPECIAL_CHARS = set("'\"<>();{}[]|&$`*%=+\\")
TEXT_COLUMN = "text"
CATEGORICAL_COLUMNS = ["method"]
NUMERIC_COLUMNS = [
    "url_length", "path_depth", "query_length", "param_count", "digit_ratio",
    "special_ratio", "status", "size", "is_client_error", "has_query", "ua_length",
]

# The same signature families the original heuristic used to label the dataset
# (github.com/mahendradata/web-log-keyword-heuristics).
LABELING_HEURISTIC = re.compile(
    r"(union\s+all\s+select|union\s+select|information_schema|\bor\s+1=1|sleep\(|"
    r"benchmark\(|shell_exec|call_user_func|/etc/passwd|\.\./|<script|"
    r"wp-login|wp-admin|phpmyadmin|/\.git|/\.env|/\.idea|xmlrpc\.php|robots\.txt)",
    re.IGNORECASE,
)


def _event_row(event: LogEvent) -> dict[str, object]:
    url = event.decoded_url
    length = max(len(url), 1)
    return {
        TEXT_COLUMN: f"{url} {event.user_agent}",
        "method": event.method,
        "url_length": len(url),
        "path_depth": event.path.count("/"),
        "query_length": len(event.query),
        "param_count": event.query.count("=") if event.query else 0,
        "digit_ratio": sum(c.isdigit() for c in url) / length,
        "special_ratio": sum(c in SPECIAL_CHARS for c in url) / length,
        "status": float(event.status),
        "size": math.log1p(max(event.size, 0)),
        "is_client_error": float(400 <= event.status < 500),
        "has_query": float(bool(event.query)),
        "ua_length": len(event.user_agent),
    }


def build_frame(events: Sequence[LogEvent]) -> pd.DataFrame:
    rows = [_event_row(event) for event in events]
    if not rows:
        return pd.DataFrame(columns=[TEXT_COLUMN, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS])
    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    report = classification_report(
        y_true, predicted, labels=[0, 1], target_names=["benign", "attack"],
        output_dict=True, zero_division=0,
    )
    return {
        "precision": float(report["attack"]["precision"]),
        "recall": float(report["attack"]["recall"]),
        "f1": float(report["attack"]["f1-score"]),
        "false_positives": int(((predicted == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((predicted == 0) & (y_true == 1)).sum()),
    }


def regex_baseline(events: Sequence[LogEvent], y_true: np.ndarray) -> dict[str, float]:
    """Approximation of the heuristic that labeled the dataset."""
    predicted = np.array(
        [1 if LABELING_HEURISTIC.search(event.decoded_url) else 0 for event in events]
    )
    return _metrics(y_true, predicted)


def supervised_baseline(
    train_events: Sequence[LogEvent],
    y_train: np.ndarray,
    test_events: Sequence[LogEvent],
    y_test: np.ndarray,
    min_precision: float = 0.90,
) -> dict[str, float]:
    """Logistic regression over char n-gram TF-IDF plus numeric features.

    Trained to reproduce the dataset's labels. The resulting metrics measure
    agreement with the labeling heuristic, not detection ability — which is exactly
    the point the numbers serve to demonstrate.
    """
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "text",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5),
                                max_features=20_000, min_df=3, sublinear_tf=True,
                            ),
                            TEXT_COLUMN,
                        ),
                        ("method", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
                        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )
    pipeline.fit(build_frame(train_events), y_train)

    train_scores = pipeline.predict_proba(build_frame(train_events))[:, 1]
    precision, _recall, thresholds = precision_recall_curve(y_train, train_scores)
    viable = [thresholds[i] for i in range(len(thresholds)) if precision[i] >= min_precision]
    threshold = float(min(viable)) if viable else 0.5

    test_scores = pipeline.predict_proba(build_frame(test_events))[:, 1]
    result = _metrics(y_test, (test_scores >= threshold).astype(int))
    result["threshold"] = threshold
    return result
