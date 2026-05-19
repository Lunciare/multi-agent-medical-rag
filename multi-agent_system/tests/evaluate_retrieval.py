import argparse
import sys
import os
import json
import random
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, SIMILARITY_TOP_K, MAX_L2_DISTANCE

RANDOM_BASELINE_SEED = 42

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}


def _load_split(split):
    filename = SPLIT_TO_FILENAME[split]
    data_path = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_smoke_test():
    print("Running Smoke Test...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "golden_dataset.json")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Smoke Test Failed: Could not load dataset. {e}")
        sys.exit(1)

    if len(dataset) != 100:
        print(f"Smoke Test Failed: Expected 100 cases, got {len(dataset)}")
        sys.exit(1)

    domain_counts = {"cardiologist": 0, "endocrinologist": 0}
    ids = set()
    tier_counts = defaultdict(lambda: defaultdict(int))

    required_keys = {"id", "tier", "tier_label", "query", "expected_specialist", "expected_keywords"}

    for case in dataset:
        missing = required_keys - set(case.keys())
        if missing:
            print(f"Smoke Test Failed: Case {case.get('id', 'UNKNOWN')} missing keys: {missing}")
            sys.exit(1)

        case_id = case["id"]
        if case_id in ids:
            print(f"Smoke Test Failed: Duplicate ID found: {case_id}")
            sys.exit(1)
        ids.add(case_id)

        tier = case["tier"]
        tier_label = case["tier_label"]
        spec = case["expected_specialist"]

        if tier not in [1, 2, 3]:
            print(f"Smoke Test Failed: Invalid tier {tier} in case {case_id}")
            sys.exit(1)

        if tier_label not in ["core", "peripheral", "out_of_scope"]:
            print(f"Smoke Test Failed: Invalid tier_label {tier_label} in case {case_id}")
            sys.exit(1)

        if spec not in domain_counts:
            print(f"Smoke Test Failed: Invalid specialist {spec} in case {case_id}")
            sys.exit(1)

        domain_counts[spec] += 1
        tier_counts[spec][tier] += 1

    if domain_counts["cardiologist"] != 50 or domain_counts["endocrinologist"] != 50:
        print(f"Smoke Test Failed: Expected 50/50 domain split, got {domain_counts}")
        sys.exit(1)

    expected_tiers = {
        "cardiologist": {1: 27, 2: 14, 3: 9},
        "endocrinologist": {1: 27, 2: 16, 3: 7}
    }

    for domain in expected_tiers:
        for t in [1, 2, 3]:
            if tier_counts[domain][t] != expected_tiers[domain][t]:
                print(f"Smoke Test Failed: Expected {domain} Tier {t} count {expected_tiers[domain][t]}, got {tier_counts[domain][t]}")
                sys.exit(1)

    print("Smoke Test Passed! Dataset is valid and correctly formatted.")
    sys.exit(0)

def _precision_at_k(docs, keywords):
    """Fraction of retrieved chunks that contain at least one expected keyword."""
    if not docs:
        return 0.0
    chunk_hits = sum(
        1 for doc in docs
        if any(kw.lower() in doc.page_content.lower() for kw in keywords)
    )
    return chunk_hits / len(docs)


def evaluate_retrieval(split="test"):
    print(f"Initializing components for retrieval evaluation (split={split})...")
    dataset = _load_split(split)

    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator (FAISS indices missing?): {e}")
        return

    total_queries = len(dataset)

    domain_hits = {"cardiologist": 0, "endocrinologist": 0}
    domain_precision_sum = {"cardiologist": 0.0, "endocrinologist": 0.0}
    domain_total = {"cardiologist": 0, "endocrinologist": 0}
    domain_hits_rand = {"cardiologist": 0, "endocrinologist": 0}
    domain_precision_rand_sum = {"cardiologist": 0.0, "endocrinologist": 0.0}

    tier_hits = defaultdict(int)
    tier_precision_sum = defaultdict(float)
    tier_hits_rand = defaultdict(int)
    tier_precision_rand_sum = defaultdict(float)
    tier_totals = defaultdict(int)
    tier_labels = {}
    tier3_results = []

    domain_pool = {
        "cardiologist": list(orchestrator.cardiologist.vectorstore.docstore._dict.values()),
        "endocrinologist": list(orchestrator.endocrinologist.vectorstore.docstore._dict.values()),
    }
    rng = random.Random(RANDOM_BASELINE_SEED)

    print(f"\nRunning retrieval evaluation on {total_queries} queries"
          f" (random baseline seed={RANDOM_BASELINE_SEED})...\n")

    for case in dataset:
        query = case["query"]
        expected_agent = case["expected_specialist"]
        keywords = case["expected_keywords"]
        tier = case.get("tier", 1)
        tier_label = case.get("tier_label", "core")

        tier_labels[tier] = tier_label

        agent = None
        if expected_agent == "cardiologist":
            agent = orchestrator.cardiologist
        elif expected_agent == "endocrinologist":
            agent = orchestrator.endocrinologist
        else:
            print(f"  [SKIP] Unknown expected agent: {expected_agent}")
            continue

        domain_total[expected_agent] += 1
        tier_totals[(expected_agent, tier)] += 1

        print(f"Query [{case['id']}]: {query[:60]}...")

        if not agent:
            print("  -> Agent not initialized. Skipping.")
            continue

        docs_and_scores = agent.vectorstore.similarity_search_with_score(query, k=SIMILARITY_TOP_K)
        retrieved_docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]

        if tier == 3:
            chunk_count = len(retrieved_docs)
            flag = "! ADJACENT CONTENT" if chunk_count > 0 else "(expected)"
            tier3_results.append((case["id"], chunk_count, flag))

        retrieved_text = " ".join(doc.page_content.lower() for doc in retrieved_docs)

        hit_found = False
        matched_words = []
        for kw in keywords:
            if kw.lower() in retrieved_text:
                hit_found = True
                matched_words.append(kw)

        chunk_hits = sum(
            1 for doc in retrieved_docs
            if any(kw.lower() in doc.page_content.lower() for kw in keywords)
        )
        precision_at_k = chunk_hits / len(retrieved_docs) if retrieved_docs else 0.0

        if hit_found:
            print(f"V  HIT (Matched keywords: {', '.join(matched_words)})"
                  f"  | P@{SIMILARITY_TOP_K}={precision_at_k:.2f}")
            domain_hits[expected_agent] += 1
            tier_hits[(expected_agent, tier)] += 1
        else:
            print(f"X MISS  | P@{SIMILARITY_TOP_K}={precision_at_k:.2f}")

        domain_precision_sum[expected_agent] += precision_at_k
        tier_precision_sum[(expected_agent, tier)] += precision_at_k

        pool = domain_pool[expected_agent]
        k = min(SIMILARITY_TOP_K, len(pool))
        random_docs = rng.sample(pool, k) if k > 0 else []
        random_text = " ".join(d.page_content.lower() for d in random_docs)
        random_hit = any(kw.lower() in random_text for kw in keywords)
        random_precision = _precision_at_k(random_docs, keywords)

        if random_hit:
            domain_hits_rand[expected_agent] += 1
            tier_hits_rand[(expected_agent, tier)] += 1
        domain_precision_rand_sum[expected_agent] += random_precision
        tier_precision_rand_sum[(expected_agent, tier)] += random_precision

    total_hits = sum(domain_hits.values())
    total_hits_rand = sum(domain_hits_rand.values())
    total_precision = sum(domain_precision_sum.values())
    total_precision_rand = sum(domain_precision_rand_sum.values())
    overall_rate = total_hits / total_queries if total_queries > 0 else 0
    overall_rate_rand = total_hits_rand / total_queries if total_queries > 0 else 0
    overall_precision = total_precision / total_queries if total_queries > 0 else 0
    overall_precision_rand = total_precision_rand / total_queries if total_queries > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"  Retrieval Evaluation Results (FAISS vs. Random Baseline, K={SIMILARITY_TOP_K})")
    print(f"{'=' * 80}")
    print(f"  {'Domain':<18} {'FAISS Hit':>11} {'FAISS P@K':>11} {'Rand Hit':>11} {'Rand P@K':>11}")
    print(f"  {'-'*18} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for domain in ("cardiologist", "endocrinologist"):
        t = domain_total[domain]
        hit_rate = domain_hits[domain] / t if t > 0 else 0
        p_at_k = domain_precision_sum[domain] / t if t > 0 else 0
        hit_rate_rand = domain_hits_rand[domain] / t if t > 0 else 0
        p_at_k_rand = domain_precision_rand_sum[domain] / t if t > 0 else 0
        print(f"  {domain:<18} {hit_rate:>10.1%} {p_at_k:>10.1%} {hit_rate_rand:>10.1%} {p_at_k_rand:>10.1%}")

    print(f"  {'-'*18} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")
    print(f"  {'OVERALL':<18} {overall_rate:>10.1%} {overall_precision:>10.1%} "
          f"{overall_rate_rand:>10.1%} {overall_precision_rand:>10.1%}")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 95}")
    print(f"  Retrieval Metrics — By Tier (FAISS vs. Random)")
    print(f"{'=' * 95}")
    print(f"  {'Domain':<18} {'Tier':<5} {'Label':<13} "
          f"{'FAISS Hit':>11} {'FAISS P@K':>11} {'Rand Hit':>11} {'Rand P@K':>11}")
    print(f"  {'-'*18} {'-'*5} {'-'*13} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            tot = tier_totals[(domain, t)]
            if tot > 0:
                hr = tier_hits[(domain, t)] / tot
                pk = tier_precision_sum[(domain, t)] / tot
                hr_r = tier_hits_rand[(domain, t)] / tot
                pk_r = tier_precision_rand_sum[(domain, t)] / tot
                print(f"  {domain:<18} {t:<5} {tier_labels.get(t, 'unknown'):<13} "
                      f"{hr:>10.1%} {pk:>10.1%} {hr_r:>10.1%} {pk_r:>10.1%}")
    print(f"{'=' * 95}")

    if tier3_results:
        print(f"\n{'=' * 60}")
        print(f"  Tier 3 (Out-of-Scope) — Fallback Behaviour")
        print(f"{'=' * 60}")
        for case_id, count, flag in tier3_results:
            print(f"  {case_id:<15} Chunks retrieved: {count:<2} {flag}")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        evaluate_retrieval(split=args.split)
