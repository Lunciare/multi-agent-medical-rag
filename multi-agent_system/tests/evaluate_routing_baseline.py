#!/usr/bin/env python3
"""Evaluate non-LLM routing baselines against a golden split.

Two baselines:
  - keyword_route: hand-curated cardiology keyword dictionary (rules-based).
  - tfidf_route:   TF-IDF (1-2 grams) + LogisticRegression trained on
                   `golden_dev.json` via `train_tfidf_router.py` (Stage 15).

Both are run on the chosen split (`--split dev|test|all`, default `test`) and
the per-domain / per-tier / overall accuracies are printed alongside Wilson
95% CIs computed via `tests._stats.fmt`.

The script also runs both baselines on `ambiguous_cases.json` so the §4.2 table
in `report_final.md` can be filled in with a TF-IDF column. The LLM Router's
predictions on the same ambiguous cases live in `evaluate_routing.py`'s output;
they are not re-derived here.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests._stats import fmt as _fmt


CARDIO_KEYWORDS = {
    'chest pain', 'palpitation', 'dyspnea', 'syncope', 'edema',
    'murmur', 'gallop', 'jugular', 'claudication', 'orthopnea',
    'cardiac', 'cardiomyopathy', 'arrhythmia', 'fibrillation', 'tachycardia',
    'bradycardia', 'stemi', 'nstemi', 'myocardial', 'infarction', 'angina',
    'ischemia', 'coronary', 'aortic', 'mitral', 'stenosis', 'regurgitation',
    'endocarditis', 'pericarditis', 'tamponade', 'heart failure', 'hfpef',
    'hcm', 'hypertrophic', 'dissection',
    'ecg', 'electrocardiogram', 'echocardiogram', 'holter', 'angiography',
    'ejection fraction', 'st elevation', 'st depression',
    'heart', 'ventricle', 'atrial', 'atrium', 'pericardial',
    'ankle-brachial', 'peripheral artery',
}

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}


def keyword_route(query):
    q = query.lower()
    for kw in CARDIO_KEYWORDS:
        if kw in q:
            return 'cardiologist'
    return 'endocrinologist'


_tfidf_pipeline = None


def _load_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "tfidf_router.pkl")
        if not os.path.exists(pickle_path):
            sys.stderr.write(
                f"ERROR: TF-IDF pickle missing at {pickle_path}. "
                f"Run `python tests/train_tfidf_router.py` first.\n"
            )
            sys.exit(2)
        with open(pickle_path, "rb") as f:
            _tfidf_pipeline = pickle.load(f)
    return _tfidf_pipeline


def tfidf_route(query):
    pipe = _load_tfidf()
    return pipe.predict([query])[0]


def _run_baseline(name, route_fn, cases):
    correct = 0
    per_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    per_tier = defaultdict(lambda: {"correct": 0, "total": 0})
    misses = []
    for c in cases:
        pred = route_fn(c["query"])
        expected = c["expected_specialist"]
        ok = pred == expected
        correct += int(ok)
        per_domain[expected]["total"] += 1
        per_domain[expected]["correct"] += int(ok)
        tier = c.get("tier", 1)
        per_tier[(expected, tier)]["total"] += 1
        per_tier[(expected, tier)]["correct"] += int(ok)
        if not ok:
            misses.append({"id": c["id"], "expected": expected, "predicted": pred,
                           "tier": tier, "query": c["query"][:60]})
    return {
        "name": name,
        "correct": correct,
        "total": len(cases),
        "per_domain": dict(per_domain),
        "per_tier": dict(per_tier),
        "misses": misses,
    }


def _print_method_table(results, *, header):
    print(f"\n{'=' * 92}")
    print(f"  {header}")
    print(f"{'=' * 92}")
    print(f"  {'Method':<20} {'Cardiology [Wilson 95% CI]':<32} "
          f"{'Endocrinology [Wilson 95% CI]':<32} {'Overall':<30}")
    print(f"  {'-'*20} {'-'*32} {'-'*32} {'-'*30}")
    for r in results:
        c = r["per_domain"].get("cardiologist", {"correct": 0, "total": 0})
        e = r["per_domain"].get("endocrinologist", {"correct": 0, "total": 0})
        print(f"  {r['name']:<20} {_fmt(c['correct'], c['total']):<32} "
              f"{_fmt(e['correct'], e['total']):<32} "
              f"{_fmt(r['correct'], r['total']):<30}")
    print(f"{'=' * 92}")


def _print_tier_table(results):
    print(f"\n{'=' * 100}")
    print(f"  Per-Tier Accuracy (Wilson 95% CI)")
    print(f"{'=' * 100}")
    domains = ("cardiologist", "endocrinologist")
    for r in results:
        print(f"\n  --- {r['name']} ---")
        print(f"  {'Domain':<18} {'Tier':<4} {'Correct':>7} {'Total':>5}  "
              f"{'Accuracy [Wilson 95% CI]':<32}")
        print(f"  {'-'*18} {'-'*4} {'-'*7} {'-'*5}  {'-'*32}")
        for dom in domains:
            for t in (1, 2, 3):
                key = (dom, t)
                if key in r["per_tier"]:
                    stats = r["per_tier"][key]
                    print(f"  {dom:<18} {t:<4} {stats['correct']:>7} {stats['total']:>5}  "
                          f"{_fmt(stats['correct'], stats['total']):<32}")
    print(f"{'=' * 100}")


def _print_misses(results):
    for r in results:
        if not r["misses"]:
            continue
        print(f"\n--- {r['name']} misses ({len(r['misses'])}) ---")
        for m in r["misses"]:
            print(f"  [{m['id']}] expected={m['expected']:<16} got={m['predicted']:<16} "
                  f"tier={m['tier']}  query={m['query']}...")


def _print_ambiguous(cases):
    print(f"\n{'=' * 100}")
    print(f"  Ambiguous Cases ({len(cases)})")
    print(f"{'=' * 100}")
    print(f"  {'ID':<10} {'Label':<55} {'Keyword':<18} {'TF-IDF':<18}")
    print(f"  {'-'*10} {'-'*55} {'-'*18} {'-'*18}")
    for c in cases:
        kw = keyword_route(c["query"])
        tf = tfidf_route(c["query"])
        print(f"  {c['id']:<10} {c['label'][:53]:<55} {kw:<18} {tf:<18}")
    print(f"{'=' * 100}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test",
                        help="Which golden split to evaluate against (default: test).")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(base_dir, 'data', SPLIT_TO_FILENAME[args.split])
    ambig_path = os.path.join(base_dir, 'data', 'ambiguous_cases.json')

    with open(golden_path, 'r') as f:
        golden = json.load(f)

    print(f"=== ROUTING BASELINE EVALUATION ({args.split.upper()}, n={len(golden)}) ===")
    results = [
        _run_baseline("Keyword Baseline", keyword_route, golden),
        _run_baseline("TF-IDF Baseline",  tfidf_route,  golden),
    ]

    _print_method_table(results,
                        header=f"Overall Routing Accuracy on the {args.split} split")
    _print_tier_table(results)
    _print_misses(results)

    if os.path.exists(ambig_path):
        with open(ambig_path, 'r') as f:
            ambiguous = json.load(f)
        _print_ambiguous(ambiguous)


if __name__ == '__main__':
    main()
