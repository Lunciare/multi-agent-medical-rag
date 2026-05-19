import argparse
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, client, AGENT_MODEL, YANDEX_PROJECT_ID, SIMILARITY_TOP_K, MAX_L2_DISTANCE

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}

def judge_faithfulness(query, context, generated_answer):
    judge_system_prompt = (
        "You are an expert Clinical QA Evaluator. Your task is to determine "
        "if a Generated Answer introduces medical facts that are NOT supported "
        "by the Retrieved Context.\n\n"
        "JUDGMENT RULES:\n"
        "1. Mark as FAITHFUL if the answer:\n"
        "   - Paraphrases or summarizes information from the Context.\n"
        "   - Uses clinical synonyms for terms in the Context "
        "(e.g., 'heart attack' for 'myocardial infarction').\n"
        "   - Draws a direct logical inference clearly supported by the Context.\n"
        "   - States 'Insufficient evidence' or declines to answer.\n"
        "2. Mark as HALLUCINATION only if the answer introduces:\n"
        "   - Specific drug names, dosages, or treatment protocols NOT in the Context.\n"
        "   - Specific lab values, thresholds, or diagnostic criteria NOT in the Context.\n"
        "   - Statistics, percentages, or epidemiological facts NOT in the Context.\n"
        "   - A specific diagnosis NOT mentioned or clearly implied by the Context.\n\n"
        "Respond ONLY with the single word 'FAITHFUL' or 'HALLUCINATION'."
    )

    judge_user_prompt = (
        f"--- CLINICAL QUERY ---\n{query}\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context}\n\n"
        f"--- GENERATED ANSWER ---\n{generated_answer}\n\n"
        "Is the Generated Answer faithful to the Retrieved Context, or does it "
        "hallucinate specific medical facts not present in the Context?"
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
        return "FAITHFUL" in judgement
    except Exception as e:
        print(f"Judge API error: {e}")
        return False


def evaluate_generation(split="test"):
    print(f"Initializing components for generation evaluation (Faithfulness, split={split})...")
    data_path = os.path.join(os.path.dirname(__file__), "data", SPLIT_TO_FILENAME[split])

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator: {e}")
        return

    total_queries = len(dataset)

    domain_faithful = {"cardiologist": 0, "endocrinologist": 0}
    domain_total = {"cardiologist": 0, "endocrinologist": 0}

    from collections import defaultdict
    tier_faithful = defaultdict(int)
    tier_totals = defaultdict(int)
    tier_labels = {}
    tier3_fallbacks = 0
    tier3_total = 0

    print(f"\nRunning generation evaluation on {total_queries} queries...\n")

    for case in dataset:
        query = case["query"]
        expected_agent = case["expected_specialist"]

        tier = case.get("tier", 1)
        tier_label = case.get("tier_label", "core")
        tier_labels[tier] = tier_label

        agent = None
        if expected_agent == "cardiologist":
            agent = orchestrator.cardiologist
        elif expected_agent == "endocrinologist":
            agent = orchestrator.endocrinologist
        else:
            continue

        print(f"Query [{case['id']}]: {query[:60]}...")

        docs_and_scores = agent.vectorstore.similarity_search_with_score(query, k=SIMILARITY_TOP_K)
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join([doc.page_content for doc in docs])

        _specialist, answer, _evidence = agent.answer(query)

        if tier == 3:
            tier3_total += 1
            if "Insufficient evidence" in answer:
                tier3_fallbacks += 1
                print("SKIP (Tier 3 'Insufficient evidence' fallback triggered)")
                continue

        domain_total[expected_agent] += 1
        tier_totals[(expected_agent, tier)] += 1

        is_faithful = judge_faithfulness(query, context, answer)

        if is_faithful:
            print("PASS (Faithful to Context)")
            domain_faithful[expected_agent] += 1
            tier_faithful[(expected_agent, tier)] += 1
        else:
            print("FAIL (Hallucination Detected!)")

    total_faithful = sum(domain_faithful.values())
    total_eval = sum(domain_total.values())
    overall_rate = total_faithful / total_eval if total_eval > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  Generation Evaluation Results (Faithfulness)")
    print(f"{'=' * 60}")
    print(f"  {'Domain':<20} {'Faithful':>8} {'Total':>6} {'Score':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")

    for domain in ("cardiologist", "endocrinologist"):
        f = domain_faithful[domain]
        t = domain_total[domain]
        rate = f / t if t > 0 else 0
        print(f"  {domain:<20} {f:>8} {t:>6} {rate:>9.1%}")

    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
    print(f"  {'OVERALL':<20} {total_faithful:>8} {total_eval:>6} {overall_rate:>9.1%}")
    print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"  Faithfulness — By Tier")
    print(f"{'=' * 60}")
    print(f"  {'Domain':<20} {'Tier':<6} {'Label':<13} {'Faithful':>8} {'Total':>6}  {'Faithfulness':>12}")
    print(f"  {'-'*20} {'-'*6} {'-'*13} {'-'*8} {'-'*6}  {'-'*12}")

    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            if tier_totals[(domain, t)] > 0:
                f = tier_faithful[(domain, t)]
                tot = tier_totals[(domain, t)]
                rate = f / tot
                print(f"  {domain:<20} {t:<6} {tier_labels.get(t, 'unknown'):<13} {f:>8} {tot:>6}  {rate:>11.1%}")
    print(f"{'=' * 60}")

    if tier3_total > 0:
        print(f"\n{'=' * 60}")
        print(f"  Tier 3 Fallback Responses")
        print(f"{'=' * 60}")
        print(f"  {tier3_fallbacks} / {tier3_total} cases returned 'Insufficient evidence' message (expected behaviour).")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    evaluate_generation(split=args.split)
