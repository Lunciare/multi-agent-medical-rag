import argparse
import csv
import math
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import AGENT_REGISTRY
from logging_config import configure_logging
from orchestrator import MedicalOrchestrator
from settings import (
    AGENT_MODEL,
    DEFAULT_KNOWLEDGE_BASE_DIR,
    MAX_L2_DISTANCE,
    PRIMARY_JUDGE_PROVIDER,
    SECONDARY_JUDGE_PROVIDER,
    SIMILARITY_TOP_K,
    TERTIARY_JUDGE_PROVIDER,
    YANDEX_PROJECT_ID,
    client,
)
from judges import JUDGE_SYSTEM_PROMPT, JudgeConfig, JudgeStats, judge_faithfulness
from tests._stats import fmt as _fmt, wilson_ci

SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}


def _legacy_judge_faithfulness(query, context, generated_answer):
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
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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


def _load_dataset(split: str):
    data_path = os.path.join(os.path.dirname(__file__), "data", SPLIT_TO_FILENAME[split])
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_judge_configs() -> list[JudgeConfig]:
    if not SECONDARY_JUDGE_PROVIDER:
        sys.stderr.write(
            "ERROR: SECONDARY_JUDGE_PROVIDER is not configured.\n"
            "Multi-judge faithfulness evaluation requires at least one judge in addition\n"
            "to the primary Yandex judge. Set the env var in .env, e.g.:\n"
            "\n"
            "  SECONDARY_JUDGE_PROVIDER=yandex:gpt://${YANDEX_PROJECT_ID}/yandexgpt-lite/latest\n"
            "  # or, if you have OpenRouter access:\n"
            "  # SECONDARY_JUDGE_PROVIDER=openrouter:meta-llama/llama-3.1-8b-instruct:free\n"
            "  # OPENROUTER_API_KEY=sk-or-v1-...\n"
            "\n"
            "Optional: TERTIARY_JUDGE_PROVIDER for a third judge.\n"
            "To bypass and run the original single-Yandex-judge evaluation:\n"
            "  python tests/evaluate_generation.py --mode yandex_only\n"
        )
        sys.exit(2)
    judges = [JudgeConfig.from_uri("yandex_primary", PRIMARY_JUDGE_PROVIDER),
              JudgeConfig.from_uri("secondary", SECONDARY_JUDGE_PROVIDER)]
    if TERTIARY_JUDGE_PROVIDER:
        judges.append(JudgeConfig.from_uri("tertiary", TERTIARY_JUDGE_PROVIDER))
    return judges


def _wilson_ci(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    from statsmodels.stats.proportion import proportion_confint
    rate = successes / total
    lo, hi = proportion_confint(successes, total, alpha=0.05, method="wilson")
    return rate, lo, hi


def _fmt_rate_ci(successes: int, total: int) -> str:
    if total == 0:
        return "—"
    rate, lo, hi = _wilson_ci(successes, total)
    return f"{rate*100:.1f}% ({successes}/{total}) [{lo*100:.1f}%–{hi*100:.1f}%]"


def _evaluate_multi_judge(split: str):
    judges = _resolve_judge_configs()
    print("Configured judges:")
    for j in judges:
        print(f"  - {j.name:<16}  provider={j.provider:<11}  model={j.model_id}")

    dataset = _load_dataset(split)
    print(f"\nInitializing components for generation evaluation (Faithfulness, "
          f"split={split}, mode=multi_judge, n={len(dataset)})...")
    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator: {e}")
        sys.exit(1)

    stats_by_judge = {j.name: JudgeStats() for j in judges}
    per_case_rows = []
    tier3_total = 0
    tier3_fallbacks = 0

    start_t = time.time()
    total_judge_calls = 0

    for case in dataset:
        query = case["query"]
        expected_agent = case["expected_specialist"]
        tier = case.get("tier", 1)

        agent = orchestrator.agents.get(expected_agent)
        if agent is None:
            continue

        print(f"\nQuery [{case['id']}]: {query[:60]}...")

        docs_and_scores = agent.vectorstore.similarity_search_with_score(
            query, k=SIMILARITY_TOP_K
        )
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join(doc.page_content for doc in docs)

        _spec, answer, _evidence = agent.answer(query)

        if tier == 3:
            tier3_total += 1
            if "Insufficient evidence" in answer:
                tier3_fallbacks += 1
                print("SKIP (Tier 3 'Insufficient evidence' fallback triggered)")
                row = {"id": case["id"], "tier": tier, "domain": expected_agent,
                       "fallback": True}
                for j in judges:
                    row[j.name] = "FALLBACK"
                per_case_rows.append(row)
                continue

        row = {"id": case["id"], "tier": tier, "domain": expected_agent,
               "fallback": False}
        for j in judges:
            verdict = judge_faithfulness(query, context, answer, j,
                                         case_id=case["id"],
                                         stats=stats_by_judge[j.name])
            total_judge_calls += 1
            if verdict is True:
                row[j.name] = "FAITHFUL"
                print(f"  [{j.name}] FAITHFUL")
            elif verdict is False:
                row[j.name] = "HALLUCINATION"
                print(f"  [{j.name}] HALLUCINATION")
            else:
                row[j.name] = "NONE"
                print(f"  [{j.name}] NONE (call failed)")
        per_case_rows.append(row)

    elapsed_s = time.time() - start_t
    _write_outputs(judges, per_case_rows, stats_by_judge, split,
                   elapsed_s=elapsed_s, total_judge_calls=total_judge_calls,
                   tier3_total=tier3_total, tier3_fallbacks=tier3_fallbacks)


def _judge_label_to_bool(label: str) -> Optional[bool]:
    if label == "FAITHFUL":
        return True
    if label == "HALLUCINATION":
        return False
    return None


def _write_outputs(judges, per_case_rows, stats_by_judge, split,
                   elapsed_s, total_judge_calls, tier3_total, tier3_fallbacks):
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports",
    )
    os.makedirs(reports_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(reports_dir, f"faithfulness_multijudge_raw_{date_str}.csv")
    md_path = os.path.join(reports_dir, f"faithfulness_multijudge_{date_str}.md")

    fieldnames = ["id", "tier", "domain", "fallback"] + [j.name for j in judges]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_case_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    judged_rows = [r for r in per_case_rows if not r.get("fallback")]

    per_judge_summary = {}
    for j in judges:
        labels = [_judge_label_to_bool(r[j.name]) for r in judged_rows]
        non_none = [v for v in labels if v is not None]
        faithful = sum(1 for v in non_none if v is True)
        per_judge_summary[j.name] = {
            "config": j,
            "total_judged": len(non_none),
            "faithful": faithful,
            "none_count": sum(1 for v in labels if v is None),
        }

    pair_kappas = []
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        cohen_kappa_score = None

    pairs = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            pairs.append((judges[i], judges[k]))

    for ja, jb in pairs:
        ax = [_judge_label_to_bool(r[ja.name]) for r in judged_rows]
        bx = [_judge_label_to_bool(r[jb.name]) for r in judged_rows]
        paired = [(a, b) for a, b in zip(ax, bx) if a is not None and b is not None]
        n = len(paired)
        if cohen_kappa_score is None or n == 0:
            kappa = None
            agree = 0
            ac1 = float("nan") if n == 0 else _gwet_ac1(paired)
        else:
            kappa = cohen_kappa_score([a for a, _ in paired], [b for _, b in paired])
            agree = sum(1 for a, b in paired if a == b)
            ac1 = _gwet_ac1(paired)
        pair_kappas.append({
            "a": ja.name, "b": jb.name, "n": n, "agreements": agree,
            "kappa": kappa, "ac1": ac1,
        })

    min_eligible = [r for r in judged_rows
                    if all(_judge_label_to_bool(r[j.name]) is not None for j in judges)]
    min_faithful = sum(
        1 for r in min_eligible
        if all(_judge_label_to_bool(r[j.name]) is True for j in judges)
    )
    excluded_for_min = len(judged_rows) - len(min_eligible)
    min_rate, min_lo, min_hi = _wilson_ci(min_faithful, len(min_eligible))

    lines = [
        f"# Multi-Judge Faithfulness Evaluation ({split} split, n={len(per_case_rows)})",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Elapsed:** {elapsed_s:.1f}s ({elapsed_s/60:.1f} min)  ",
        f"**Total judge calls:** {total_judge_calls}  ",
        f"**Raw per-case CSV:** [`{os.path.basename(csv_path)}`]({os.path.basename(csv_path)})",
        "",
        "## Configured Judges",
        "",
        "| Role | Provider | Model URI | Auth | Rate-limit | Conn | Timeout | Other | Successes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for j in judges:
        s = stats_by_judge[j.name]
        lines.append(
            f"| {j.name} | {j.provider} | `{j.model_id}` | "
            f"{s.auth_errors} | {s.rate_limit_errors} | {s.connection_errors} | "
            f"{s.timeout_errors} | {s.other_errors} | {s.successes} |"
        )

    lines += [
        "",
        "## Per-Judge Faithfulness (test split, Wilson 95% CI)",
        "",
        "| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for j in judges:
        s = per_judge_summary[j.name]
        rate, lo, hi = _wilson_ci(s["faithful"], s["total_judged"])
        lines.append(
            f"| {j.name} | {s['faithful']} | {s['total_judged']} | {s['none_count']} | "
            f"{rate*100:.1f}% | [{lo*100:.1f}%–{hi*100:.1f}%] |"
        )

    lines += [
        "",
        "## Minimum-Judge Faithfulness (all judges = FAITHFUL)",
        "",
        f"Cases where every available judge returned a non-None label: {len(min_eligible)} / {len(judged_rows)} "
        f"(excluded {excluded_for_min} due to None from at least one judge).  ",
        f"All judges agreed FAITHFUL on **{min_faithful} / {len(min_eligible)}** cases = "
        f"**{min_rate*100:.1f}% [Wilson 95% CI {min_lo*100:.1f}%–{min_hi*100:.1f}%]**.",
        "",
        "## Pairwise Agreement (Cohen's κ and Gwet's AC1)",
        "",
        "Cohen's κ is degenerate when one rater's marginal class probability is 0 "
        "(i.e. labels everything the same way), since chance agreement equals observed "
        "agreement and the κ denominator vanishes. Gwet's AC1 stays well-defined in that "
        "regime: it uses the empirical class prior (averaged across raters) to compute "
        "chance agreement, which only vanishes when both raters tie on the same extreme. "
        "Report both; AC1 is the better-behaved chance-corrected statistic when one "
        "marginal collapses.",
        "",
        "| Pair | n (both non-None) | Agreements | Cohen's κ | Gwet's AC1 | Landis & Koch (on AC1) |",
        "|---|---|---|---|---|---|",
    ]
    label_counts = {j.name: {True: 0, False: 0} for j in judges}
    for row in judged_rows:
        for j in judges:
            v = _judge_label_to_bool(row[j.name])
            if v is not None:
                label_counts[j.name][v] += 1
    for pk in pair_kappas:
        kappa_str = "n/a" if pk["kappa"] is None else f"{pk['kappa']:.3f}"
        ac1_str = "n/a" if pk["ac1"] is None or math.isnan(pk["ac1"]) else f"{pk['ac1']:.3f}"
        lk = _landis_koch(
            pk["kappa"],
            both_labels_seen_a=sum(1 for v, c in label_counts[pk["a"]].items() if c > 0),
            both_labels_seen_b=sum(1 for v, c in label_counts[pk["b"]].items() if c > 0),
            n=pk["n"], agreements=pk["agreements"],
            ac1=pk["ac1"],
        )
        lines.append(
            f"| ({pk['a']}, {pk['b']}) | {pk['n']} | {pk['agreements']} | {kappa_str} | {ac1_str} | {lk} |"
        )

    disagreements = _collect_disagreements(judges, judged_rows)
    lines += [
        "",
        "## Disagreement Cases",
        "",
        f"{len(disagreements)} case(s) where the configured judges did not all agree on a "
        f"non-None label (excluding fallbacks).",
        "",
        "| Case | Tier | Domain | " + " | ".join(j.name for j in judges) + " |",
        "|---|---|---|" + "---|" * len(judges),
    ]
    for row in disagreements:
        cells = " | ".join(row[j.name] for j in judges)
        lines.append(f"| {row['id']} | {row['tier']} | {row['domain']} | {cells} |")

    if tier3_total:
        lines += [
            "",
            "## Tier 3 Fallback",
            "",
            f"{tier3_fallbacks} / {tier3_total} Tier 3 cases returned the "
            f"'Insufficient evidence' fallback (excluded from judge totals).",
        ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n{'='*60}")
    print(f"  Multi-Judge Faithfulness Evaluation Complete")
    print(f"{'='*60}")
    print(f"Raw CSV:        {csv_path}")
    print(f"Summary report: {md_path}")
    for j in judges:
        s = per_judge_summary[j.name]
        rate, lo, hi = _wilson_ci(s["faithful"], s["total_judged"])
        print(f"  {j.name:<18} FAITHFUL {s['faithful']}/{s['total_judged']} = "
              f"{rate*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]  (None: {s['none_count']})")
    print(f"  {'MINIMUM':<18} FAITHFUL {min_faithful}/{len(min_eligible)} = "
          f"{min_rate*100:.1f}% [{min_lo*100:.1f}%-{min_hi*100:.1f}%]"
          f"  (excluded {excluded_for_min} due to None)")
    for pk in pair_kappas:
        ks = "n/a" if pk["kappa"] is None else f"{pk['kappa']:.3f}"
        a1 = "n/a" if pk["ac1"] is None or math.isnan(pk["ac1"]) else f"{pk['ac1']:.3f}"
        print(f"  ({pk['a']},{pk['b']}): κ={ks}  Gwet AC1={a1}  n={pk['n']}")
    print(f"Elapsed: {elapsed_s:.1f}s ({elapsed_s/60:.1f} min); "
          f"total judge calls: {total_judge_calls}")


def _gwet_ac1(pairs: list) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    n_true = sum((1 if a else 0) + (1 if b else 0) for a, b in pairs)
    pi_true = n_true / (2 * n)
    p_e = 2 * pi_true * (1 - pi_true)
    if 1 - p_e == 0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _landis_koch_band(coeff: float) -> str:
    if coeff is None or math.isnan(coeff):
        return "undefined"
    if coeff < 0.0:
        return "less than chance"
    if coeff < 0.4:
        return "poor"
    if coeff < 0.6:
        return "moderate"
    if coeff < 0.8:
        return "substantial"
    return "almost perfect"


def _landis_koch(kappa: float, *, both_labels_seen_a: int, both_labels_seen_b: int,
                 n: int, agreements: int, ac1: float = float("nan")) -> str:
    if both_labels_seen_a < 2 or both_labels_seen_b < 2:
        ac1_band = _landis_koch_band(ac1)
        return (
            f"degenerate κ (one marginal = 0; observed agreement = "
            f"{agreements}/{n}); via AC1: {ac1_band}"
        )
    if kappa is None or math.isnan(kappa):
        return "degenerate (κ undefined)"
    return _landis_koch_band(kappa)


def _collect_disagreements(judges, judged_rows):
    disagreements = []
    for r in judged_rows:
        labels = [r[j.name] for j in judges]
        non_fallback_set = set(l for l in labels if l != "FALLBACK")
        if len(non_fallback_set) > 1:
            disagreements.append(r)
    return disagreements


def _evaluate_yandex_only(split: str):
    print("Initializing components for generation evaluation "
          f"(Faithfulness, split={split}, mode=yandex_only)...")
    dataset = _load_dataset(split)
    try:
        orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
    except Exception as e:
        print(f"Error loading orchestrator: {e}")
        return

    total_queries = len(dataset)
    domain_faithful = {k: 0 for k in AGENT_REGISTRY}
    domain_total = {k: 0 for k in AGENT_REGISTRY}
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

        agent = orchestrator.agents.get(expected_agent)
        if agent is None:
            continue

        print(f"Query [{case['id']}]: {query[:60]}...")
        docs_and_scores = agent.vectorstore.similarity_search_with_score(
            query, k=SIMILARITY_TOP_K
        )
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join(doc.page_content for doc in docs)
        _spec, answer, _ev = agent.answer(query)

        if tier == 3:
            tier3_total += 1
            if "Insufficient evidence" in answer:
                tier3_fallbacks += 1
                print("SKIP (Tier 3 'Insufficient evidence' fallback triggered)")
                continue

        domain_total[expected_agent] += 1
        tier_totals[(expected_agent, tier)] += 1
        is_faithful = _legacy_judge_faithfulness(query, context, answer)
        if is_faithful:
            print("PASS (Faithful to Context)")
            domain_faithful[expected_agent] += 1
            tier_faithful[(expected_agent, tier)] += 1
        else:
            print("FAIL (Hallucination Detected!)")

    total_faithful = sum(domain_faithful.values())
    total_eval = sum(domain_total.values())
    print(f"\n{'='*80}")
    print(f"  Generation Evaluation Results (Faithfulness, yandex_only; Wilson 95% CI)")
    print(f"{'='*80}")
    print(f"  {'Domain':<20} {'Faithful':>8} {'Total':>6}  {'Score [Wilson 95% CI]':<30}")
    print(f"  {'-'*20} {'-'*8} {'-'*6}  {'-'*30}")
    for domain in sorted(AGENT_REGISTRY.keys()):
        f = domain_faithful[domain]
        t = domain_total[domain]
        print(f"  {domain:<20} {f:>8} {t:>6}  {_fmt(f, t):<30}")
    print(f"  {'-'*20} {'-'*8} {'-'*6}  {'-'*30}")
    print(f"  {'OVERALL':<20} {total_faithful:>8} {total_eval:>6}  "
          f"{_fmt(total_faithful, total_eval):<30}")
    print(f"{'='*80}")
    if tier3_total > 0:
        print(f"\n  {tier3_fallbacks} / {tier3_total} Tier 3 cases returned "
              f"'Insufficient evidence' fallback "
              f"({_fmt(tier3_fallbacks, tier3_total)}).")


def evaluate_generation(split="test", mode="multi_judge"):
    if mode == "yandex_only":
        _evaluate_yandex_only(split)
    elif mode == "multi_judge":
        _evaluate_multi_judge(split)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--mode", choices=["multi_judge", "yandex_only"],
                        default="multi_judge",
                        help="multi_judge (default) uses primary + secondary (+ optional tertiary) "
                             "judges and writes a CSV/markdown summary; yandex_only is the "
                             "backward-compatible single-judge mode.")
    args = parser.parse_args()
    evaluate_generation(split=args.split, mode=args.mode)
