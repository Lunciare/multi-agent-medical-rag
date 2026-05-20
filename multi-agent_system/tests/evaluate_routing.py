#!/usr/bin/env python3
import argparse
import sys
import os
import json
import time
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import client, ROUTING_MODEL, YANDEX_PROJECT_ID
from tests._stats import fmt as _fmt

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}


ROUTING_SYSTEM_PROMPT = (
    "You are a medical orchestrator. "
    "Determine which specialist should handle the request: "
    "cardiologist or endocrinologist. "
    "Respond strictly in one word."
)


def route_query(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=10,
            extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"__error__:{e}"


SPECIALIST_ALIASES = {
    "cardiologist":     "cardiologist",
    "cardiology":       "cardiologist",
    "endocrinologist":  "endocrinologist",
    "endocrinology":    "endocrinologist",
}


def normalise(raw: str) -> str:
    raw_clean = raw.strip().lower().rstrip(".")
    if raw_clean in SPECIALIST_ALIASES:
        return SPECIALIST_ALIASES[raw_clean]
    for key, canonical in SPECIALIST_ALIASES.items():
        if key in raw_clean:
            return canonical
    return raw_clean


def evaluate_routing(split="test"):
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    golden_path = os.path.join(data_dir, SPLIT_TO_FILENAME[split])
    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    ambiguous_path = os.path.join(data_dir, "ambiguous_cases.json")
    ambiguous_cases = []
    if os.path.exists(ambiguous_path):
        with open(ambiguous_path, "r", encoding="utf-8") as f:
            ambiguous_cases = json.load(f)

    total = len(dataset)
    print(f"Running routing evaluation on {total} golden queries (split={split})…\n")

    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0, "details": []})
    tier_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    tier_labels = {}
    overall_correct = 0

    for case in dataset:
        qid = case["id"]
        query = case["query"]
        expected = case["expected_specialist"]
        tier = case.get("tier", 1)
        tier_label = case.get("tier_label", "core")
        tier_labels[tier] = tier_label

        raw_response = route_query(query)
        predicted = normalise(raw_response)

        is_correct = predicted == expected
        if is_correct:
            overall_correct += 1
            domain_stats[expected]["correct"] += 1
            tier_stats[(expected, tier)]["correct"] += 1
        domain_stats[expected]["total"] += 1
        tier_stats[(expected, tier)]["total"] += 1

        mark = "V" if is_correct else "X"
        extra = f"  (raw: {raw_response})" if predicted != raw_response.strip().lower() else ""
        print(f"  {mark} [{qid}]  expected={expected}  got={predicted}{extra}")

        domain_stats[expected]["details"].append({
            "id": qid, "expected": expected, "predicted": predicted,
            "raw": raw_response, "correct": is_correct,
        })
        time.sleep(0.3)

    overall_acc = overall_correct / total if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  Routing Evaluation — Golden Dataset (Wilson 95% CI)")
    print(f"{'=' * 70}")
    print(f"  {'Domain':<20} {'Correct':>8} {'Total':>8}  {'Accuracy [Wilson 95% CI]':<28}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}  {'-'*28}")

    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        print(f"  {domain:<20} {s['correct']:>8} {s['total']:>8}  {_fmt(s['correct'], s['total']):<28}")

    print(f"  {'-'*20} {'-'*8} {'-'*8}  {'-'*28}")
    print(f"  {'OVERALL':<20} {overall_correct:>8} {total:>8}  {_fmt(overall_correct, total):<28}")
    print(f"{'=' * 70}\n")

    print(f"{'=' * 80}")
    print(f"  Routing Accuracy — By Tier (Wilson 95% CI)")
    print(f"{'=' * 80}")
    print(f"  {'Domain':<20} {'Tier':<6} {'Label':<13} {'Correct':>7} {'Total':>7}  {'Accuracy [Wilson 95% CI]':<28}")
    print(f"  {'-'*20} {'-'*6} {'-'*13} {'-'*7} {'-'*7}  {'-'*28}")

    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            if tier_stats[(domain, t)]["total"] > 0:
                c = tier_stats[(domain, t)]["correct"]
                tot = tier_stats[(domain, t)]["total"]
                print(f"  {domain:<20} {t:<6} {tier_labels.get(t, 'unknown'):<13} "
                      f"{c:>7} {tot:>7}  {_fmt(c, tot):<28}")
    print(f"{'=' * 80}\n")

    ambiguous_details = []
    if ambiguous_cases:
        print(f"{'=' * 60}")
        print(f"  Cross-Domain Ambiguous Cases ({len(ambiguous_cases)} queries)")
        print(f"{'=' * 60}\n")

        for case in ambiguous_cases:
            qid = case["id"]
            query = case["query"]
            domains = case["domains"]
            label = case["label"]

            raw_response = route_query(query)
            predicted = normalise(raw_response)
            in_expected = predicted in domains

            flag = "AMBIGUOUS"
            symbol = "<->" if in_expected else "?"
            print(f"  {symbol} [{qid}]  routed_to={predicted}  "
                  f"valid_domains={domains}  ({flag}: {label})")

            ambiguous_details.append({
                "id": qid, "label": label, "predicted": predicted,
                "raw": raw_response, "valid_domains": domains,
                "in_expected": in_expected,
            })
            time.sleep(0.3)

        print(f"\n{'=' * 60}\n")

    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports",
    )
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"routing_evaluation_{timestamp}.md")

    lines = [
        "# Routing Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Golden Dataset — Accuracy (Wilson 95% CI)",
        "",
        "| Domain | Correct | Total | Accuracy [Wilson 95% CI] |",
        "|---|---|---|---|",
    ]
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        lines.append(f"| {domain} | {s['correct']} | {s['total']} | {_fmt(s['correct'], s['total'])} |")
    lines.append(f"| **Overall** | **{overall_correct}** | **{total}** | **{_fmt(overall_correct, total)}** |")

    lines += [
        "",
        "## Golden Dataset — Routing Accuracy By Tier (Wilson 95% CI)",
        "",
        "| Domain | Tier | Label | Correct | Total | Accuracy [Wilson 95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for domain in ("cardiologist", "endocrinologist"):
        for t in [1, 2, 3]:
            if tier_stats[(domain, t)]["total"] > 0:
                c = tier_stats[(domain, t)]["correct"]
                tot = tier_stats[(domain, t)]["total"]
                lines.append(f"| {domain} | {t} | {tier_labels.get(t, 'unknown')} | {c} | {tot} | {_fmt(c, tot)} |")

    lines += [
        "",
        "## Golden Dataset — Per-Query Details",
        "",
        "| ID | Expected | Predicted | Raw LLM Output | Result |",
        "|---|---|---|---|---|",
    ]
    for domain in sorted(domain_stats.keys()):
        for d in domain_stats[domain]["details"]:
            mark = "V" if d["correct"] else "X"
            lines.append(
                f"| {d['id']} | {d['expected']} | {d['predicted']} "
                f"| {d['raw']} | {mark} |"
            )

    if ambiguous_details:
        lines += [
            "",
            "## Cross-Domain Ambiguous Cases",
            "",
            "These queries intentionally span multiple medical domains. "
            "No single routing decision is considered \"correct\" — the table "
            "documents observed behaviour.",
            "",
            "| ID | Label | Routed To | Valid Domains | In Valid? |",
            "|---|---|---|---|---|",
        ]
        for d in ambiguous_details:
            in_flag = "V" if d["in_expected"] else "?"
            lines.append(
                f"| {d['id']} | {d['label']} | {d['predicted']} "
                f"| {', '.join(d['valid_domains'])} | {in_flag} |"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    evaluate_routing(split=args.split)
