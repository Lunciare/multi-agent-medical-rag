#!/usr/bin/env python3
"""Grid-search the RefusalGate thresholds on golden_dev.json.

Outputs:
  - reports/refusal_gate_grid.csv      — full grid: (signal, threshold, T3 recall, T1/2 FP rate, F1).
  - settings.py (updated)              — writes REFUSAL_GATE_SIGNAL and L2_REJECT_MIN (Signal A). Signal B's chosen `corpus_dist_k` is reported on stdout but no longer persisted to settings (post-Stage-32 cleanup); pass it into `RefusalGate(corpus_dist_k=...)` at construction time.
  - stdout                             — precision/recall table and the chosen signal.

Tunes against `golden_dev.json`, which by construction contains the 30 development
cases (`cardio_1..15`, `endo_1..15`). Only 1 of those is Tier 3 (`cardio_10` —
aortic dissection), so the "≥80% T3 rejection on dev" target reduces to "reject
cardio_10". To avoid that single-sample bias, the tuner also reports each
threshold's performance on the test-split's 15 T3 cases as a confirmation read,
but uses *only* the dev numbers to pick the threshold.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import AGENT_REGISTRY
from orchestrator import MedicalOrchestrator
from refusal_gate import (
    CorpusDistStats,
    RefusalGate,
    SIGNAL_A_MIN_L2,
    SIGNAL_B_CORPUS_K,
    load_or_compute_corpus_dist_stats,
)
from settings import (
    DEFAULT_KNOWLEDGE_BASE_DIR,
    MAX_L2_DISTANCE,
    SIMILARITY_TOP_K,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEV_PATH = os.path.join(DATA_DIR, "golden_dev.json")
TEST_PATH = os.path.join(DATA_DIR, "golden_test.json")
REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports",
)
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "settings.py")


L2_REJECT_MIN_GRID = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
                      0.99, 1.00, 1.02, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
CORPUS_DIST_K_GRID = [-3.5, -3.0, -2.5, -2.0, -1.7, -1.5, -1.3, -1.0, -0.7, -0.5,
                      -0.3, 0.0, 0.3, 0.5, 1.0]


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_min_dists(dataset, orchestrator) -> List[dict]:
    """For each case, retrieve top-K and record min L2 distance + meta."""
    rows = []
    for case in dataset:
        agent = orchestrator.agents.get(case["expected_specialist"])
        if agent is None:
            continue
        ds = agent.vectorstore.similarity_search_with_score(case["query"], k=SIMILARITY_TOP_K)
        min_dist = min((s for _d, s in ds), default=None)
        rows.append({
            "id": case["id"],
            "tier": case["tier"],
            "domain": case["expected_specialist"],
            "min_dist": float(min_dist) if min_dist is not None else None,
            "n_retrieved": len(ds),
        })
    return rows


def _confusion(rows, *, reject_fn) -> dict:
    """rows: list of {id, tier, domain, min_dist}. reject_fn(min_dist) → bool.

    Positive class = Tier 3 (out-of-scope). Negative class = Tier 1/2.
    """
    tp = fn = fp = tn = 0
    fp_ids = []
    fn_ids = []
    for r in rows:
        positive = r["tier"] == 3
        rejected = r["min_dist"] is None or reject_fn(r["min_dist"])
        if positive and rejected:
            tp += 1
        elif positive and not rejected:
            fn += 1
            fn_ids.append(r["id"])
        elif not positive and rejected:
            fp += 1
            fp_ids.append(r["id"])
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "fp_rate": fp_rate, "f1": f1,
        "fp_ids": fp_ids, "fn_ids": fn_ids,
    }


def _signal_a_reject(threshold):
    return lambda md: md > threshold


def _signal_b_reject(threshold):
    return lambda md: md > threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write-settings", action="store_true",
                        help="Skip writing REFUSAL_GATE_SIGNAL and L2_REJECT_MIN back to settings.py")
    args = parser.parse_args()

    print("Loading orchestrator + FAISS indices...")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)

    # Stage 39 four-specialist update: build the per-specialty corpus-distance
    # stats from AGENT_REGISTRY directly so the tuner auto-expands as
    # specialties are added. Cache files live in each specialty's folder
    # (matches the convention written by RefusalGate.from_vectorstore()).
    print("\nLoading corpus distance stats (per specialty)...")
    corpus_stats = {}
    for key, cfg in AGENT_REGISTRY.items():
        cache_path = os.path.join(cfg["folder_path"], "corpus_dist_stats.json")
        corpus_stats[key] = load_or_compute_corpus_dist_stats(
            orchestrator.agents[key].vectorstore,
            cache_path=cache_path,
            specialty=key,
        )
        print(f"  {key:<22}: μ={corpus_stats[key].mu:.4f}  "
              f"σ={corpus_stats[key].sigma:.4f}")

    print("\nCollecting min_dist per case (dev + test for confirmation)...")
    dev = _load(DEV_PATH)
    test = _load(TEST_PATH)
    dev_rows = _collect_min_dists(dev, orchestrator)
    test_rows = _collect_min_dists(test, orchestrator)

    n_dev_t3 = sum(1 for r in dev_rows if r["tier"] == 3)
    n_dev_neg = sum(1 for r in dev_rows if r["tier"] in (1, 2))
    n_test_t3 = sum(1 for r in test_rows if r["tier"] == 3)
    n_test_neg = sum(1 for r in test_rows if r["tier"] in (1, 2))
    print(f"  dev:  Tier3 (pos)={n_dev_t3}, Tier1/2 (neg)={n_dev_neg}")
    print(f"  test: Tier3 (pos)={n_test_t3}, Tier1/2 (neg)={n_test_neg}")

    grid_rows = []
    print(f"\n=== Signal A (min L2 distance) — grid over L2_REJECT_MIN ===")
    print(f"  {'thresh':>7}  {'dev TP':>7}  {'dev FN':>7}  {'dev FP':>7}  "
          f"{'dev TN':>7}  {'dev recall':>10}  {'dev FP rate':>11}  {'test TP':>7}  "
          f"{'test FP rate':>12}")
    best_a = None
    for thr in L2_REJECT_MIN_GRID:
        rej = _signal_a_reject(thr)
        cm_dev = _confusion(dev_rows, reject_fn=rej)
        cm_test = _confusion(test_rows, reject_fn=rej)
        print(f"  {thr:>7.3f}  {cm_dev['tp']:>7}  {cm_dev['fn']:>7}  "
              f"{cm_dev['fp']:>7}  {cm_dev['tn']:>7}  "
              f"{cm_dev['recall']:>10.1%}  {cm_dev['fp_rate']:>11.1%}  "
              f"{cm_test['tp']:>7}  {cm_test['fp_rate']:>12.1%}")
        grid_rows.append({
            "signal": "A",
            "param_name": "L2_REJECT_MIN",
            "param_value": thr,
            "dev_tp": cm_dev["tp"], "dev_fn": cm_dev["fn"],
            "dev_fp": cm_dev["fp"], "dev_tn": cm_dev["tn"],
            "dev_recall": cm_dev["recall"], "dev_fp_rate": cm_dev["fp_rate"],
            "dev_precision": cm_dev["precision"], "dev_f1": cm_dev["f1"],
            "dev_fp_ids": ";".join(cm_dev["fp_ids"]),
            "dev_fn_ids": ";".join(cm_dev["fn_ids"]),
            "test_tp": cm_test["tp"], "test_fn": cm_test["fn"],
            "test_fp": cm_test["fp"], "test_tn": cm_test["tn"],
            "test_recall": cm_test["recall"], "test_fp_rate": cm_test["fp_rate"],
            "test_precision": cm_test["precision"], "test_f1": cm_test["f1"],
            "test_fp_ids": ";".join(cm_test["fp_ids"]),
            "test_fn_ids": ";".join(cm_test["fn_ids"]),
        })
        if (cm_dev["recall"] >= 0.8 and cm_dev["fp_rate"] <= 0.05 and
                (best_a is None or thr < best_a["threshold"])):
            best_a = {"threshold": thr, "dev": cm_dev, "test": cm_test}

    print(f"\n=== Signal B (μ_corpus − k·σ_corpus) — grid over CORPUS_DIST_K ===")
    print("  Threshold computed per-domain using each specialty's μ, σ.")
    print(f"  {'k':>6}  {'thr-cardio':>10}  {'thr-endo':>10}  {'dev TP':>7}  "
          f"{'dev FN':>7}  {'dev FP':>7}  {'dev TN':>7}  "
          f"{'dev recall':>10}  {'dev FP rate':>11}  {'test TP':>7}  "
          f"{'test FP rate':>12}")
    best_b = None
    for k in CORPUS_DIST_K_GRID:
        # Stage 39: per-specialty threshold built from AGENT_REGISTRY.
        thr_by_domain = {
            specialty: stats.mu - k * stats.sigma
            for specialty, stats in corpus_stats.items()
        }
        # Pretty-print only the first two domain thresholds in the table header
        # (the historical 2-column layout); the full per-specialty thresholds
        # are reachable by reading `corpus_stats` / the per-specialty
        # `corpus_dist_stats.json` files.
        thr_cardio = thr_by_domain.get("cardiologist", float("nan"))
        thr_endo = thr_by_domain.get("endocrinologist", float("nan"))

        def make_b_reject(thr_by_domain):
            def fn(row):
                return row["min_dist"] is None or row["min_dist"] > thr_by_domain[row["domain"]]
            return fn
        rej_fn = make_b_reject(thr_by_domain)

        def _confusion_per_row(rows):
            tp = fn = fp = tn = 0
            fp_ids = []
            fn_ids = []
            for r in rows:
                positive = r["tier"] == 3
                rejected = rej_fn(r)
                if positive and rejected:
                    tp += 1
                elif positive and not rejected:
                    fn += 1; fn_ids.append(r["id"])
                elif not positive and rejected:
                    fp += 1; fp_ids.append(r["id"])
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "precision": precision, "recall": recall, "fp_rate": fp_rate, "f1": f1,
                    "fp_ids": fp_ids, "fn_ids": fn_ids}
        cm_dev = _confusion_per_row(dev_rows)
        cm_test = _confusion_per_row(test_rows)
        print(f"  {k:>6.2f}  {thr_cardio:>10.4f}  {thr_endo:>10.4f}  "
              f"{cm_dev['tp']:>7}  {cm_dev['fn']:>7}  {cm_dev['fp']:>7}  "
              f"{cm_dev['tn']:>7}  {cm_dev['recall']:>10.1%}  {cm_dev['fp_rate']:>11.1%}  "
              f"{cm_test['tp']:>7}  {cm_test['fp_rate']:>12.1%}")
        grid_rows.append({
            "signal": "B",
            "param_name": "CORPUS_DIST_K",
            "param_value": k,
            "dev_tp": cm_dev["tp"], "dev_fn": cm_dev["fn"],
            "dev_fp": cm_dev["fp"], "dev_tn": cm_dev["tn"],
            "dev_recall": cm_dev["recall"], "dev_fp_rate": cm_dev["fp_rate"],
            "dev_precision": cm_dev["precision"], "dev_f1": cm_dev["f1"],
            "dev_fp_ids": ";".join(cm_dev["fp_ids"]),
            "dev_fn_ids": ";".join(cm_dev["fn_ids"]),
            "test_tp": cm_test["tp"], "test_fn": cm_test["fn"],
            "test_fp": cm_test["fp"], "test_tn": cm_test["tn"],
            "test_recall": cm_test["recall"], "test_fp_rate": cm_test["fp_rate"],
            "test_precision": cm_test["precision"], "test_f1": cm_test["f1"],
            "test_fp_ids": ";".join(cm_test["fp_ids"]),
            "test_fn_ids": ";".join(cm_test["fn_ids"]),
        })
        if (cm_dev["recall"] >= 0.8 and cm_dev["fp_rate"] <= 0.05 and
                (best_b is None or k > best_b["k"])):
            best_b = {"k": k, "dev": cm_dev, "test": cm_test}

    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "refusal_gate_grid.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if grid_rows:
            writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grid_rows)
    print(f"\nGrid saved to {csv_path}")

    # ----- choose signal -----
    print(f"\n=== Selection (dev-only target: recall ≥ 80% AND FP rate ≤ 5%) ===")
    print(f"  Signal A best: {best_a}")
    print(f"  Signal B best: {best_b}")

    # Prefer Signal A if it meets the target; otherwise Signal B; otherwise the closest A entry.
    if best_a is not None:
        chosen_signal = "A"
        chosen_l2 = best_a["threshold"]
        chosen_k = 0.0
        print(f"  → CHOSEN: Signal A  L2_REJECT_MIN={chosen_l2:.3f}  "
              f"(dev recall {best_a['dev']['recall']:.1%}, dev FP rate {best_a['dev']['fp_rate']:.1%}, "
              f"test recall {best_a['test']['recall']:.1%}, test FP rate {best_a['test']['fp_rate']:.1%})")
    elif best_b is not None:
        chosen_signal = "B"
        chosen_l2 = 1.20  # keep last good Signal A as fallback constant in settings
        chosen_k = best_b["k"]
        print(f"  → CHOSEN: Signal B  CORPUS_DIST_K={chosen_k:.2f}  "
              f"(dev recall {best_b['dev']['recall']:.1%}, dev FP rate {best_b['dev']['fp_rate']:.1%}, "
              f"test recall {best_b['test']['recall']:.1%}, test FP rate {best_b['test']['fp_rate']:.1%})")
    else:
        # No entry hits the target. Pick the best F1 row over both grids.
        best_row = max(grid_rows, key=lambda r: r["dev_f1"])
        chosen_signal = best_row["signal"]
        if chosen_signal == "A":
            chosen_l2 = best_row["param_value"]
            chosen_k = 0.0
        else:
            chosen_l2 = 1.20
            chosen_k = best_row["param_value"]
        print(f"  → No threshold meets the strict target; falling back to best F1 row.")
        print(f"  → CHOSEN: Signal {chosen_signal}  "
              f"param={best_row['param_value']}  "
              f"(dev recall {best_row['dev_recall']:.1%}, dev FP rate {best_row['dev_fp_rate']:.1%}, "
              f"test recall {best_row['test_recall']:.1%}, test FP rate {best_row['test_fp_rate']:.1%})")

    # ----- write back to settings.py -----
    if not args.no_write_settings:
        _update_settings(chosen_signal=chosen_signal,
                         l2_reject_min=chosen_l2)
        print(f"\nUpdated {SETTINGS_PATH}:")
        print(f"  REFUSAL_GATE_SIGNAL    = {chosen_signal!r}")
        print(f"  L2_REJECT_MIN          = {chosen_l2}")
        if chosen_signal == 'B':
            print(f"  (Signal B chosen with CORPUS_DIST_K={chosen_k:.3f}; "
                  "this value is reported here but no longer persisted to "
                  "settings.py — pass it into RefusalGate(corpus_dist_k=...) "
                  "at construction time. See refusal_gate.py.)")
    else:
        print("\n(skipping settings.py write because --no-write-settings was passed)")


def _update_settings(*, chosen_signal: str, l2_reject_min: float):
    """Append/replace REFUSAL_GATE_SIGNAL and L2_REJECT_MIN in settings.py.

    CORPUS_DIST_K is intentionally no longer written here — the Stage 32
    cleanup removed it from settings.py because Signal A is the production
    runtime path. If a future Signal-B re-tune chooses a non-default
    `corpus_dist_k`, pass it into `RefusalGate(corpus_dist_k=...)` at
    construction time rather than re-adding the settings constant.
    """
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    sentinel = "# --- refusal-gate threshold (managed by tests/tune_refusal_gate.py) ---"
    block = (
        f"\n{sentinel}\n"
        f"# Signal A min-L2 threshold. See report_final.md §4.5 for the trade-off\n"
        f"# against Tier 1/2 false-positive rate; full analysis in Stage 7 report.\n"
        f"REFUSAL_GATE_SIGNAL = {chosen_signal!r}\n"
        f"L2_REJECT_MIN = {float(l2_reject_min):.3f}\n"
    )
    if sentinel in text:
        head, _, _ = text.partition(sentinel)
        text = head.rstrip() + block
    else:
        text = text.rstrip() + "\n" + block
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
