#!/usr/bin/env python3
import argparse
import os
import sys
import json
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR


def tune_retrieval():
    data_path = os.path.join(os.path.dirname(__file__), "data", "golden_dev.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading orchestrator for grid search…")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)

    K_VALUES = [3, 5, 7, 10, 15]
    L2_THRESHOLDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]

    CHOSEN_K = 5
    CHOSEN_L2 = 1.2

    rows = []

    header = f"  {'K':<4} | {'L2 ≤':<6} | {'Hits':>5} | {'Hit Rate':>9} | {'Avg Ret.':>8} | {'Note'}"
    sep = f"  {'-'*4}-+-{'-'*6}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*12}"

    print(f"\n{sep}")
    print(header)
    print(sep)

    best_score = -1
    best_params = {}

    for k in K_VALUES:
        for l2_max in L2_THRESHOLDS:
            hits = 0
            total_retrieved = 0

            for case in dataset:
                query = case["query"]
                agent_type = case["expected_specialist"]
                keywords = case["expected_keywords"]

                agent = (orchestrator.cardiologist if agent_type == "cardiologist"
                         else orchestrator.endocrinologist)

                docs_and_scores = agent.vectorstore.similarity_search_with_score(query, k=k)
                valid_docs = [doc for doc, score in docs_and_scores if score <= l2_max]

                total_retrieved += len(valid_docs)

                if valid_docs:
                    context = " ".join(doc.page_content.lower() for doc in valid_docs)
                    if any(kw.lower() in context for kw in keywords):
                        hits += 1

            hit_rate = hits / len(dataset) * 100
            avg_ret = total_retrieved / len(dataset)

            is_chosen = (k == CHOSEN_K and l2_max == CHOSEN_L2)
            note = "◀ CHOSEN" if is_chosen else ""

            if hit_rate > best_score or (hit_rate == best_score and avg_ret < best_params.get("avg_ret", 999)):
                best_score = hit_rate
                best_params = {"k": k, "l2": l2_max, "avg_ret": avg_ret}
                if not is_chosen:
                    note = "★ best" if note == "" else note + " ★ best"

            print(f"  {k:<4} | {l2_max:<6.1f} | {hits:>5} | {hit_rate:>8.1f}% | {avg_ret:>8.1f} | {note}")

            rows.append({
                "K": k,
                "L2_threshold": l2_max,
                "hits": hits,
                "total_queries": len(dataset),
                "hit_rate_pct": round(hit_rate, 1),
                "avg_docs_retrieved": round(avg_ret, 2),
                "is_chosen": is_chosen,
            })

    print(sep)
    print(f"\n  Best: K={best_params['k']}, L2 ≤ {best_params['l2']:.1f} "
          f"(Hit Rate {best_score:.1f}%, Avg {best_params['avg_ret']:.1f} docs)")
    print(f"  Chosen operating point: K={CHOSEN_K}, L2 ≤ {CHOSEN_L2}")

    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports",
    )
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "hyperparameter_grid.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "K", "L2_threshold", "hits", "total_queries",
            "hit_rate_pct", "avg_docs_retrieved", "is_chosen",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n📄 Grid saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="dev",
        help="Tuning is locked to the dev split (golden_dev.json). "
             "Any value other than 'dev' is rejected.",
    )
    args = parser.parse_args()
    if args.split != "dev":
        sys.stderr.write(
            f"tune_retrieval.py is locked to --split dev "
            f"(got --split {args.split}). Hyperparameters must be tuned on the "
            f"dev split only; evaluate on test via evaluate_*.py --split test.\n"
        )
        sys.exit(2)
    tune_retrieval()
