import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, SIMILARITY_TOP_K, MAX_L2_DISTANCE

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

def evaluate_retrieval():
    print("Initializing components for retrieval evaluation...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "golden_dataset.json")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator (FAISS indices missing?): {e}")
        return

    total_queries = len(dataset)

    domain_hits = {"cardiologist": 0, "endocrinologist": 0}
    domain_total = {"cardiologist": 0, "endocrinologist": 0}

    tier_hits = defaultdict(int)
    tier_totals = defaultdict(int)
    tier_labels = {}
    tier3_results = []

    print(f"\nRunning retrieval evaluation on {total_queries} queries...\n")

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
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]

        if tier == 3:
            chunk_count = len(docs)
            flag = "⚠️ ADJACENT CONTENT" if chunk_count > 0 else "(expected)"
            tier3_results.append((case["id"], chunk_count, flag))

        retrieved_text = " ".join([doc.page_content.lower() for doc in docs])

        hit_found = False
        matched_words = []
        for kw in keywords:
            if kw.lower() in retrieved_text:
                hit_found = True
                matched_words.append(kw)

        if hit_found:
            print(f"HIT (Matched keywords: {', '.join(matched_words)})")
            domain_hits[expected_agent] += 1
            tier_hits[(expected_agent, tier)] += 1
        else:
            print("MISS (None of the expected keywords found in retrieved context)")

    total_hits = sum(domain_hits.values())
    overall_rate = total_hits / total_queries if total_queries > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  Retrieval Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  {'Domain':<20} {'Hits':>6} {'Total':>6} {'Hit Rate':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*10}")

    for domain in ("cardiologist", "endocrinologist"):
        h = domain_hits[domain]
        t = domain_total[domain]
        rate = h / t if t > 0 else 0
        print(f"  {domain:<20} {h:>6} {t:>6} {rate:>9.1%}")

    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*10}")
    print(f"  {'OVERALL':<20} {total_hits:>6} {total_queries:>6} {overall_rate:>9.1%}")
    print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"  Retrieval Hit Rate — By Tier")
    print(f"{'=' * 60}")
    print(f"  {'Domain':<20} {'Tier':<6} {'Label':<13} {'Hits':>6} {'Total':>6}  {'Hit Rate':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*13} {'-'*6} {'-'*6}  {'-'*10}")

    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            if tier_totals[(domain, t)] > 0:
                h = tier_hits[(domain, t)]
                tot = tier_totals[(domain, t)]
                rate = h / tot
                print(f"  {domain:<20} {t:<6} {tier_labels.get(t, 'unknown'):<13} {h:>6} {tot:>6}  {rate:>9.1%}")
    print(f"{'=' * 60}")

    if tier3_results:
        print(f"\n{'=' * 60}")
        print(f"  Tier 3 (Out-of-Scope) — Fallback Behaviour")
        print(f"{'=' * 60}")
        for case_id, count, flag in tier3_results:
            print(f"  {case_id:<15} Chunks retrieved: {count:<2} {flag}")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        run_smoke_test()
    else:
        evaluate_retrieval()
