import argparse
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, client, AGENT_MODEL, YANDEX_PROJECT_ID, SIMILARITY_TOP_K, MAX_L2_DISTANCE
from tests._stats import fmt as _fmt

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}

def judge_context_relevance(query, context):
    judge_system_prompt = (
        "You are an expert Medical Context Evaluator. Your task is to evaluate "
        "if the Provided Context contains sufficient, highly relevant medical "
        "evidence, treatment protocols, or diagnostic criteria to comprehensively "
        "answer the User Query.\n\n"
        "If the context is just a bag of isolated keywords, unrelated case reports, "
        "or lacking specific medical protocol required by the query, output ONLY 'INSUFFICIENT'.\n"
        "If the context contains the actual medical logic needed to formulate an accurate answer, output ONLY 'SUFFICIENT'."
    )

    judge_user_prompt = (
        f"--- CLINICAL QUERY ---\n{query}\n\n"
        f"--- PROVIDED CONTEXT ---\n{context}\n"
    )

    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": judge_user_prompt},
            ],
            temperature=0.0,
            max_tokens=64,
            extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
        )
        judgement = response.choices[0].message.content.strip().upper()
        return "SUFFICIENT" in judgement
    except Exception as e:
        print(f"Judge API error: {e}")
        return False

def evaluate_relevance(split="test"):
    print(f"Initializing components for chunk Context Relevancy evaluation (split={split})...")
    data_path = os.path.join(os.path.dirname(__file__), "data", SPLIT_TO_FILENAME[split])

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator: {e}")
        return

    total_queries = len(dataset)

    domain_sufficient = {"cardiologist": 0, "endocrinologist": 0}
    domain_total = {"cardiologist": 0, "endocrinologist": 0}

    from collections import defaultdict
    tier_sufficient = defaultdict(int)
    tier_totals = defaultdict(int)
    tier_labels = {}

    print(f"\nRunning context relevancy evaluation on {total_queries} queries...\n")

    for case in dataset:
        query = case["query"]
        expected_agent = case["expected_specialist"]

        tier = case.get("tier", 1)
        tier_label = case.get("tier_label", "core")
        tier_labels[tier] = tier_label

        agent = None
        if expected_agent == "cardiologist":
            agent = orchestrator.agents["cardiologist"]
        elif expected_agent == "endocrinologist":
            agent = orchestrator.agents["endocrinologist"]
        else:
            continue

        domain_total[expected_agent] += 1
        tier_totals[(expected_agent, tier)] += 1

        print(f"Query [{case['id']}]: {query[:60]}...")

        docs_and_scores = agent.vectorstore.similarity_search_with_score(query, k=SIMILARITY_TOP_K)
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join([doc.page_content for doc in docs])

        is_sufficient = judge_context_relevance(query, context)

        if is_sufficient:
            print("SUFFICIENT Context Retrieved")
            domain_sufficient[expected_agent] += 1
            tier_sufficient[(expected_agent, tier)] += 1
        else:
            print("INSUFFICIENT Context Retrieved (Needs optimization)")

    total_sufficient = sum(domain_sufficient.values())

    print(f"\n{'=' * 80}")
    print(f"  Chunk Relevancy Evaluation Results (Wilson 95% CI)")
    print(f"{'=' * 80}")
    print(f"  {'Domain':<20} {'Sufficient':>10} {'Total':>6}  {'Relevancy [Wilson 95% CI]':<30}")
    print(f"  {'-'*20} {'-'*10} {'-'*6}  {'-'*30}")

    for domain in ("cardiologist", "endocrinologist"):
        s = domain_sufficient[domain]
        t = domain_total[domain]
        print(f"  {domain:<20} {s:>10} {t:>6}  {_fmt(s, t):<30}")

    print(f"  {'-'*20} {'-'*10} {'-'*6}  {'-'*30}")
    print(f"  {'OVERALL':<20} {total_sufficient:>10} {total_queries:>6}  "
          f"{_fmt(total_sufficient, total_queries):<30}")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 90}")
    print(f"  Chunk Relevancy — By Tier (Wilson 95% CI)")
    print(f"{'=' * 90}")
    print(f"  {'Domain':<20} {'Tier':<6} {'Label':<13} {'Sufficient':>10} {'Total':>6}  "
          f"{'Relevancy [Wilson 95% CI]':<30}")
    print(f"  {'-'*20} {'-'*6} {'-'*13} {'-'*10} {'-'*6}  {'-'*30}")

    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            if tier_totals[(domain, t)] > 0:
                s = tier_sufficient[(domain, t)]
                tot = tier_totals[(domain, t)]
                print(f"  {domain:<20} {t:<6} {tier_labels.get(t, 'unknown'):<13} "
                      f"{s:>10} {tot:>6}  {_fmt(s, tot):<30}")
    print(f"{'=' * 90}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    evaluate_relevance(split=args.split)
