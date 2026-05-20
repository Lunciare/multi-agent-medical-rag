#!/usr/bin/env python3
"""Train a TF-IDF + Logistic Regression router on `golden_dev.json`.

Pipeline (per Stage 15 spec):
  TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)
  → LogisticRegression(C=1.0)

The trained pipeline is pickled to `tests/data/tfidf_router.pkl` and loaded by
`evaluate_routing_baseline.py:tfidf_route(query)` at evaluation time.

Training set: 30-case `golden_dev.json` (`cardio_1..15` + `endo_1..15` per
Stage 4 Fix 1). The `query` field is the input text and `expected_specialist`
is the binary label. No tuning is performed inside this script — `C=1.0` is the
fixed Stage 15 setting; future stages can hyper-tune with `tune_*` style grids
if needed.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEV_PATH = os.path.join(DATA_DIR, "golden_dev.json")
PICKLE_PATH = os.path.join(DATA_DIR, "tfidf_router.pkl")


def _load_dev_xy() -> tuple[list[str], list[str]]:
    """Return (queries, labels) from golden_dev.json."""
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    queries: list[str] = []
    labels: list[str] = []
    for c in cases:
        queries.append(c["query"])
        labels.append(c["expected_specialist"])
    return queries, labels


def build_pipeline() -> Pipeline:
    """Construct the TF-IDF + LogisticRegression pipeline used in §4.1."""
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2)),
        ("lr", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def main() -> None:
    if not os.path.exists(DEV_PATH):
        sys.stderr.write(
            f"ERROR: golden_dev.json missing at {DEV_PATH}. "
            f"Run Stage 4's data-split first.\n"
        )
        sys.exit(2)

    queries, labels = _load_dev_xy()
    print(f"Loaded {len(queries)} dev cases from {DEV_PATH}")
    n_cardio = sum(1 for y in labels if y == "cardiologist")
    n_endo = sum(1 for y in labels if y == "endocrinologist")
    print(f"  Class balance: cardiologist={n_cardio}, endocrinologist={n_endo}")

    pipe = build_pipeline()
    pipe.fit(queries, labels)
    print("Fitted Pipeline(TfidfVectorizer(ngram=(1,2), max_df=0.9, min_df=2) → "
          "LogisticRegression(C=1.0)) on the dev split.")

    train_acc = pipe.score(queries, labels)
    print(f"  Training accuracy (memorisation check, NOT generalisation): "
          f"{train_acc*100:.1f}% ({int(round(train_acc * len(queries)))}/{len(queries)})")

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(pipe, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved pipeline → {PICKLE_PATH} ({os.path.getsize(PICKLE_PATH):,} bytes)")


if __name__ == "__main__":
    main()
