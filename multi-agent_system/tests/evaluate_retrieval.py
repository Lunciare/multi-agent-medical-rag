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


def _doc_keys_from_gold(gold_sources):
    """Convert a list of gold source dicts to a set of doc identifiers.

    Matching is on doc_name when present (which is unique across the corpus);
    otherwise on source_file as a fallback. Returns a frozenset of strings.
    """
    keys = set()
    for g in gold_sources or []:
        if isinstance(g, dict):
            key = g.get("doc_name") or g.get("source_file")
            if key:
                keys.add(key)
        elif isinstance(g, str):
            keys.add(g)
    return keys


def _doc_key_of(doc):
    """Doc-level identity for a retrieved chunk. Matches the gold annotation key."""
    md = getattr(doc, "metadata", {}) or {}
    return md.get("doc_name") or md.get("source_file")


def _recall_at_k(retrieved_docs, gold_sources):
    """Fraction of gold sources that appear at least once in retrieved_docs.

    Returns None if gold_sources is empty (e.g., Tier 3 with no correct chunk,
    or an unannotated Tier 1/2 case).
    """
    gold_keys = _doc_keys_from_gold(gold_sources)
    if not gold_keys:
        return None
    retrieved_keys = {_doc_key_of(d) for d in retrieved_docs if _doc_key_of(d)}
    return len(gold_keys & retrieved_keys) / len(gold_keys)


def _mrr_at_k(retrieved_docs, gold_sources):
    """Reciprocal rank of the first retrieved gold source. 0 if none of the gold
    documents appear in the top-K. Returns None if gold_sources is empty."""
    gold_keys = _doc_keys_from_gold(gold_sources)
    if not gold_keys:
        return None
    for rank, doc in enumerate(retrieved_docs, start=1):
        if _doc_key_of(doc) in gold_keys:
            return 1.0 / rank
    return 0.0


def _wilson_ci(successes, total):
    if total == 0:
        return 0.0, 0.0, 0.0
    from statsmodels.stats.proportion import proportion_confint
    rate = successes / total
    lo, hi = proportion_confint(successes, total, alpha=0.05, method="wilson")
    return rate, lo, hi


def _load_case_by_id(case_id):
    """Find a case across any of the dev/test/all splits."""
    for split in ("all", "test", "dev"):
        for case in _load_split(split):
            if case["id"] == case_id:
                return case
    raise SystemExit(f"Case id {case_id!r} not found in any split.")


def print_sources(case_id, top_k=20):
    """Debug helper for the annotation workflow.

    Retrieves the top-K chunks for the given case, ignores MAX_L2_DISTANCE
    (so even barely-retrieved chunks are shown for context), and prints each
    chunk's metadata + first 200 chars + keyword match count, so a human
    annotator can decide which documents actually contain the answer.
    """
    case = _load_case_by_id(case_id)
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    agent = (orchestrator.cardiologist if case["expected_specialist"] == "cardiologist"
             else orchestrator.endocrinologist)
    print(f"\n=== {case_id}  tier={case['tier']}/{case['tier_label']}  domain={case['expected_specialist']} ===")
    print(f"Query: {case['query']}")
    print(f"Expected keywords: {', '.join(case['expected_keywords'])}")
    print(f"Existing gold_sources: {case.get('gold_sources', '(not annotated)')}")
    print()

    docs_and_scores = agent.vectorstore.similarity_search_with_score(case["query"], k=top_k)
    keywords_lower = [k.lower() for k in case["expected_keywords"]]
    seen_docs = {}
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        md = doc.metadata or {}
        sf = md.get("source_file", "?")
        dn = md.get("doc_name", "?")
        cat = md.get("category", "?")
        within_threshold = score <= MAX_L2_DISTANCE
        flag = "" if within_threshold else "  (BEYOND L2_THRESHOLD)"
        text_lower = doc.page_content.lower()
        kw_hits = [kw for kw in keywords_lower if kw in text_lower]
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{rank:>2}] L2={score:.3f}  cat={cat:<10}  source_file={sf}{flag}")
        print(f"     doc_name : {dn}")
        print(f"     kw hits  : {len(kw_hits)} of {len(keywords_lower)}  "
              f"({', '.join(kw_hits) if kw_hits else 'none'})")
        print(f"     preview  : {preview}")
        print()
        seen_docs.setdefault(dn, {"first_rank": rank, "chunks": 0, "kw_hits": set(),
                                  "source_file": sf})
        seen_docs[dn]["chunks"] += 1
        seen_docs[dn]["kw_hits"].update(kw_hits)

    print(f"=== Unique documents seen in top-{top_k} ===")
    print(f"{'doc_name':<60} {'first_rank':>10} {'chunks':>8} {'kw hits':>10}")
    for dn, info in sorted(seen_docs.items(), key=lambda x: x[1]["first_rank"]):
        print(f"{dn[:58]:<60} {info['first_rank']:>10} {info['chunks']:>8} "
              f"{len(info['kw_hits']):>10}")


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

    # New grounded metrics — gold_sources-based.
    # domain-level
    domain_recall_sum = {"cardiologist": 0.0, "endocrinologist": 0.0}
    domain_mrr_sum    = {"cardiologist": 0.0, "endocrinologist": 0.0}
    domain_recall_n   = {"cardiologist": 0,   "endocrinologist": 0}
    # tier-level (only T1/T2 contribute)
    tier_recall_sum = defaultdict(float)
    tier_mrr_sum    = defaultdict(float)
    tier_recall_n   = defaultdict(int)
    # Tier 3 refusal — fraction of T3 cases where zero chunks were retrieved.
    tier3_refusals = defaultdict(int)
    tier3_count    = defaultdict(int)
    # RefusalGate (Stage 7) — gate's refusal verdict counted per-tier/per-domain.
    gate_refusals = defaultdict(int)
    gate_totals   = defaultdict(int)

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
            tier3_count[expected_agent] += 1
            if chunk_count == 0:
                tier3_refusals[expected_agent] += 1

        gate_totals[(expected_agent, tier)] += 1
        if agent.refuse(query):
            gate_refusals[(expected_agent, tier)] += 1

        gold_sources = case.get("gold_sources")
        if tier in (1, 2):
            recall = _recall_at_k(retrieved_docs, gold_sources)
            mrr    = _mrr_at_k(retrieved_docs, gold_sources)
            if recall is not None:
                domain_recall_sum[expected_agent] += recall
                domain_mrr_sum[expected_agent]    += mrr
                domain_recall_n[expected_agent]   += 1
                tier_recall_sum[(expected_agent, tier)] += recall
                tier_mrr_sum[(expected_agent, tier)]    += mrr
                tier_recall_n[(expected_agent, tier)]   += 1

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

    # === New grounded metrics (Recall@K, MRR@K) ===
    print(f"\n{'=' * 95}")
    print(f"  Grounded Retrieval Metrics  (K={SIMILARITY_TOP_K}, against gold_sources)")
    print(f"  Recall@K — fraction of gold documents that appear in the retrieved set.")
    print(f"  MRR@K    — reciprocal rank of the first retrieved gold document.")
    print(f"  KeywordHitRate (legacy) — kept for cross-stage comparison; "
          f"see report §4.3 note.")
    print(f"{'=' * 95}")
    print(f"  {'Domain':<18} {'Recall@K':>10} {'MRR@K':>10} {'n':>6} "
          f"{'KW Hit (legacy)':>18}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*6} {'-'*18}")
    overall_recall_num = 0.0
    overall_mrr_num    = 0.0
    overall_recall_n   = 0
    for domain in ("cardiologist", "endocrinologist"):
        n = domain_recall_n[domain]
        recall = domain_recall_sum[domain] / n if n > 0 else 0
        mrr    = domain_mrr_sum[domain]    / n if n > 0 else 0
        legacy_hit = (domain_hits[domain] / domain_total[domain]
                      if domain_total[domain] > 0 else 0)
        print(f"  {domain:<18} {recall:>9.1%} {mrr:>10.3f} {n:>6} {legacy_hit:>17.1%}")
        overall_recall_num += domain_recall_sum[domain]
        overall_mrr_num    += domain_mrr_sum[domain]
        overall_recall_n   += n
    if overall_recall_n > 0:
        overall_recall = overall_recall_num / overall_recall_n
        overall_mrr    = overall_mrr_num    / overall_recall_n
    else:
        overall_recall = 0
        overall_mrr    = 0
    overall_legacy_hit = total_hits / total_queries if total_queries else 0
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*6} {'-'*18}")
    print(f"  {'OVERALL (T1+T2)':<18} {overall_recall:>9.1%} {overall_mrr:>10.3f} "
          f"{overall_recall_n:>6} {overall_legacy_hit:>17.1%}")
    print(f"{'=' * 95}")

    print(f"\n{'=' * 95}")
    print(f"  Grounded Retrieval Metrics — By Tier  (T1+T2 only; T3 reports refusal rate)")
    print(f"{'=' * 95}")
    print(f"  {'Domain':<18} {'Tier':<5} {'Label':<13} "
          f"{'Recall@K':>10} {'MRR@K':>10} {'n':>6} {'KW Hit (legacy)':>18}")
    print(f"  {'-'*18} {'-'*5} {'-'*13} {'-'*10} {'-'*10} {'-'*6} {'-'*18}")
    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2]:
            n = tier_recall_n[(domain, t)]
            if n == 0:
                continue
            recall = tier_recall_sum[(domain, t)] / n
            mrr    = tier_mrr_sum[(domain, t)]    / n
            tier_legacy = (tier_hits[(domain, t)] / tier_totals[(domain, t)]
                           if tier_totals[(domain, t)] else 0)
            print(f"  {domain:<18} {t:<5} {tier_labels.get(t, '?'):<13} "
                  f"{recall:>9.1%} {mrr:>10.3f} {n:>6} {tier_legacy:>17.1%}")
    print(f"{'=' * 95}")

    # RefusalGate (Stage 7) — separate from the legacy zero-chunk metric.
    print(f"\n{'=' * 80}")
    print(f"  RefusalGate Verdicts  (Stage 7 numeric gate, signal=A, L2_REJECT_MIN tuned)")
    print(f"  Positive class = Tier 3 (target ≥80% recall on test).")
    print(f"  Negative class = Tier 1/2 (target ≤5% false-positive rate on test).")
    print(f"{'=' * 80}")
    print(f"  {'Domain':<18} {'Tier':<5} {'Label':<13} {'Refused':>8} {'Total':>6} "
          f"{'Refusal Rate':>13}")
    print(f"  {'-'*18} {'-'*5} {'-'*13} {'-'*8} {'-'*6} {'-'*13}")
    gate_t3_refused = 0
    gate_t3_total  = 0
    gate_t12_refused = 0
    gate_t12_total  = 0
    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            tot = gate_totals[(domain, t)]
            if tot == 0:
                continue
            ref = gate_refusals[(domain, t)]
            rate = ref / tot
            label = tier_labels.get(t, "?")
            print(f"  {domain:<18} {t:<5} {label:<13} {ref:>8} {tot:>6} {rate:>12.1%}")
            if t == 3:
                gate_t3_refused += ref
                gate_t3_total  += tot
            else:
                gate_t12_refused += ref
                gate_t12_total  += tot
    print(f"  {'-'*18} {'-'*5} {'-'*13} {'-'*8} {'-'*6} {'-'*13}")
    rec = gate_t3_refused / gate_t3_total if gate_t3_total else 0
    fpr = gate_t12_refused / gate_t12_total if gate_t12_total else 0
    print(f"  {'TIER 3 RECALL':<18} {'':<5} {'':<13} "
          f"{gate_t3_refused:>8} {gate_t3_total:>6} {rec:>12.1%}")
    print(f"  {'TIER 1/2 FP RATE':<18} {'':<5} {'':<13} "
          f"{gate_t12_refused:>8} {gate_t12_total:>6} {fpr:>12.1%}")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 60}")
    print(f"  Tier 3 Refusal Rate  (legacy: zero-chunk retrieval fraction)")
    print(f"{'=' * 60}")
    print(f"  {'Domain':<18} {'Refusals':>10} {'T3 Total':>10} {'Refusal Rate':>14}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*14}")
    refusals_total = 0
    t3_total = 0
    for domain in ("cardiologist", "endocrinologist"):
        r = tier3_refusals[domain]
        tot = tier3_count[domain]
        refusals_total += r
        t3_total       += tot
        rate = r / tot if tot > 0 else 0
        print(f"  {domain:<18} {r:>10} {tot:>10} {rate:>13.1%}")
    overall_refusal_rate = (refusals_total / t3_total) if t3_total else 0
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*14}")
    print(f"  {'OVERALL':<18} {refusals_total:>10} {t3_total:>10} {overall_refusal_rate:>13.1%}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--case-id", default=None,
                        help="Show top-K retrieval for a single case (annotation workflow).")
    parser.add_argument("--print-sources", action="store_true",
                        help="With --case-id, dump retrieved sources (source_file, doc_name, kw hits).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K to retrieve when --print-sources is set (default 20).")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
    elif args.case_id and args.print_sources:
        print_sources(args.case_id, top_k=args.top_k)
    else:
        evaluate_retrieval(split=args.split)
