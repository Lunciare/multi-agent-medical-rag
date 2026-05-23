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
from logging_config import configure_logging
from tests._stats import fmt as _fmt


# Stage 39 four-specialist extension:
#   The pre-Stage-39 baseline was a binary cardio-vs-rest rule. It now
#   needs an explicit precedence between four specialty keyword sets.
#   Precedence rule: count how many distinct keywords from each specialty's
#   set appear in the query; pick the specialty with the highest count; on
#   ties, fall back to the registry order (cardiologist, endocrinologist,
#   gastroenterologist, infectionist). Endocrinology is the historical
#   default if no specialty has any keyword hit (kept for backward
#   compatibility with the Stage 1 baseline — the most common "default"
#   case in the 2-specialty world was endocrine because cardiology had a
#   richer keyword vocabulary).
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
ENDO_KEYWORDS = {
    'diabetes', 'hba1c', 'insulin', 'glucose', 'hyperglycaemia', 'hypoglycaemia',
    'thyroid', 'tsh', 'free t4', 'free t3', 'hypothyroidism', 'hyperthyroidism',
    'goitre', 'graves', 'hashimoto', 'thyroiditis', 'thyroid nodule',
    'adrenal', 'cortisol', 'cushing', 'addison', 'pheochromocytoma', 'hyperaldosteronism',
    'pituitary', 'prolactin', 'acromegaly', 'growth hormone',
    'parathyroid', 'pth', 'hypercalcaemia', 'hypocalcaemia',
    'metabolic syndrome', 'obesity', 'pcos', 'osteoporosis', 'dexa',
    'endocrine', 'gland', 'hormonal', 'hormone',
}
GASTRO_KEYWORDS = {
    'gerd', 'reflux', 'heartburn', 'peptic ulcer', 'h. pylori', 'helicobacter',
    'ulcerative colitis', "crohn's", 'crohn', 'ibd', 'inflammatory bowel',
    'irritable bowel', 'ibs', 'coeliac', 'celiac', 'gluten',
    'hepatitis b', 'hepatitis c', 'hbv', 'hcv', 'cirrhosis', 'ascites',
    'varices', 'variceal', 'liver failure', 'hepatic encephalopathy',
    'cholelithiasis', 'gallstones', 'gallbladder', 'cholangitis',
    'pancreatitis', 'pancreatic',
    'colon', 'colorectal', 'diverticul', 'haemorrhoid', 'hemorrhoid',
    'diarrhoea', 'diarrhea', 'constipation', 'melaena', 'haematochezia',
    'dysphagia', 'oesophag', 'esophag', 'achalasia', 'barrett',
    'nafld', 'masld', 'fatty liver',
    'paracentesis', 'endoscopy', 'colonoscopy',
    'gastro', 'liver', 'biliary', 'stomach', 'bowel',
}
INFECT_KEYWORDS = {
    'pneumonia', 'sepsis', 'septic shock', 'cellulitis', 'meningitis', 'encephalitis',
    'tuberculosis', 'tb', 'hiv', 'aids', 'antiretroviral',
    'urinary tract infection', 'uti', 'pyelonephritis',
    'covid', 'sars-cov-2', 'influenza',
    'malaria', 'plasmodium', 'dengue', 'typhoid', 'leptospirosis',
    'clostridium difficile', 'c. difficile', 'cdi',
    'mrsa', 'staphylococcus aureus', 'streptococcus',
    'osteomyelitis', 'bacteraemia', 'bacteremia', 'fungaemia',
    'aspergillosis', 'cryptococcal', 'histoplasm', 'candidiasis',
    'antimicrobial', 'antibiotic', 'antiviral', 'antifungal', 'antiparasitic',
    'vaccin', 'immunisation', 'immunization',
    'parasite', 'helminth', 'protozoa',
    'fever of unknown origin', 'febrile neutropenia',
    'rifampicin', 'isoniazid', 'doxycycline', 'penicillin', 'ceftriaxone',
    'infection', 'infectious',
}

# Order used for tie-break in keyword_route — matches AGENT_REGISTRY's iteration
# order, which in turn matches the routing prompt's enumeration.
_KEYWORD_SETS_IN_REGISTRY_ORDER = [
    ('cardiologist', CARDIO_KEYWORDS),
    ('endocrinologist', ENDO_KEYWORDS),
    ('gastroenterologist', GASTRO_KEYWORDS),
    ('infectionist', INFECT_KEYWORDS),
]

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}


def keyword_route(query):
    """Highest-keyword-count specialty wins; registry order breaks ties.

    Empty hits across all four specialties → fall back to 'endocrinologist'
    (preserves the Stage 1 baseline's tie-break behaviour). The 4-specialist
    extension changes this from a binary cardio-vs-rest rule (Stage 1) to a
    multi-class counter; tie-breaks now follow registry order — see comment
    on _KEYWORD_SETS_IN_REGISTRY_ORDER above.
    """
    q = query.lower()
    best_specialty = None
    best_count = 0
    for specialty, kws in _KEYWORD_SETS_IN_REGISTRY_ORDER:
        count = sum(1 for kw in kws if kw in q)
        if count > best_count:
            best_count = count
            best_specialty = specialty
    if best_specialty is None:
        return 'endocrinologist'  # Stage 1 default
    return best_specialty


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


_REGISTRY_DOMAINS = ("cardiologist", "endocrinologist", "gastroenterologist", "infectionist")


def _print_method_table(results, *, header):
    print(f"\n{'=' * 130}")
    print(f"  {header}")
    print(f"{'=' * 130}")
    cols = "  " + f"{'Method':<20}"
    for d in _REGISTRY_DOMAINS:
        cols += f" {d+' [95% CI]':<26}"
    cols += f" {'Overall [95% CI]':<28}"
    print(cols)
    print("  " + "-"*20 + (" " + "-"*26) * len(_REGISTRY_DOMAINS) + f" {'-'*28}")
    for r in results:
        line = "  " + f"{r['name']:<20}"
        for d in _REGISTRY_DOMAINS:
            s = r["per_domain"].get(d, {"correct": 0, "total": 0})
            line += " " + f"{_fmt(s['correct'], s['total']):<26}"
        line += " " + f"{_fmt(r['correct'], r['total']):<28}"
        print(line)
    print(f"{'=' * 130}")


def _print_tier_table(results):
    print(f"\n{'=' * 100}")
    print(f"  Per-Tier Accuracy (Wilson 95% CI)")
    print(f"{'=' * 100}")
    for r in results:
        print(f"\n  --- {r['name']} ---")
        print(f"  {'Domain':<22} {'Tier':<4} {'Correct':>7} {'Total':>5}  "
              f"{'Accuracy [Wilson 95% CI]':<32}")
        print(f"  {'-'*22} {'-'*4} {'-'*7} {'-'*5}  {'-'*32}")
        for dom in _REGISTRY_DOMAINS:
            for t in (1, 2, 3):
                key = (dom, t)
                if key in r["per_tier"]:
                    stats = r["per_tier"][key]
                    print(f"  {dom:<22} {t:<4} {stats['correct']:>7} {stats['total']:>5}  "
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
    configure_logging()
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
