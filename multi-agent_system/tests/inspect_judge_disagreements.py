#!/usr/bin/env python3
"""Read-only diagnostic: print per-disagreement context for the multi-judge run.

Given the raw CSV from evaluate_generation.py --mode multi_judge, this script
re-derives each disagreement case, replays retrieval against the live FAISS
index, regenerates the answer with the same agent prompt, and writes a markdown
report that lists query / first-500-chars-of-context / answer / each judge's
verdict for each disagreement.

No new labels are produced — this is purely diagnostic output the author can
reference when writing report_final.md §5.3.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, MAX_L2_DISTANCE, SIMILARITY_TOP_K


def _load_dataset_by_id() -> dict:
    by_id = {}
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    for fn in ("golden_dataset.json", "golden_test.json", "golden_dev.json"):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for case in json.load(f):
                by_id.setdefault(case["id"], case)
    return by_id


def _is_disagreement(row: dict, judge_cols: list[str]) -> bool:
    labels = {row[c] for c in judge_cols if c in row and row[c] not in ("", "FALLBACK")}
    return len(labels) > 1


def main(csv_path: str, out_path: str | None = None) -> None:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        all_cols = reader.fieldnames or []

    judge_cols = [c for c in all_cols if c not in ("id", "tier", "domain", "fallback")]
    if not judge_cols:
        sys.stderr.write(f"No judge columns found in {csv_path}. "
                         f"Available columns: {all_cols}\n")
        sys.exit(2)

    disagreements = [r for r in rows if _is_disagreement(r, judge_cols)]
    print(f"Found {len(disagreements)} disagreement case(s) across "
          f"{len(judge_cols)} judges ({', '.join(judge_cols)}).")

    if not disagreements:
        print("No disagreements to inspect — exiting without writing a report.")
        return

    by_id = _load_dataset_by_id()
    print("Loading orchestrator to replay retrieval and generation...")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)

    lines = [
        f"# Judge Disagreement Inspection",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Source CSV:** [`{os.path.basename(csv_path)}`]({os.path.basename(csv_path)})  ",
        f"**Judges examined:** {', '.join(judge_cols)}  ",
        f"**Disagreement cases:** {len(disagreements)}",
        "",
        "Each entry below replays the retrieval and generation against the live FAISS "
        "indices and reproduces the first 500 chars of the retrieved context plus the "
        "full generated answer, alongside each judge's verdict. No new judge calls are "
        "made — verdicts are read straight from the raw CSV.",
        "",
        "---",
        "",
    ]

    for row in disagreements:
        case_id = row["id"]
        case = by_id.get(case_id)
        if case is None:
            lines += [
                f"## {case_id} — case not found in golden_*.json",
                "",
                "Could not locate the source case for this row. "
                "Listing verdicts only:",
                "",
                "| Judge | Verdict |", "|---|---|",
            ]
            for c in judge_cols:
                lines.append(f"| {c} | {row[c]} |")
            lines += ["", "---", ""]
            continue

        expected_agent = case["expected_specialist"]
        agent = (orchestrator.agents["cardiologist"] if expected_agent == "cardiologist"
                 else orchestrator.agents["endocrinologist"])
        docs_and_scores = agent.vectorstore.similarity_search_with_score(
            case["query"], k=SIMILARITY_TOP_K
        )
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join(d.page_content for d in docs)
        _spec, answer, _ev = agent.answer(case["query"])

        lines += [
            f"## {case_id} (tier={case.get('tier','?')}, domain={expected_agent})",
            "",
            "**Verdicts:**",
            "",
            "| Judge | Verdict |",
            "|---|---|",
        ]
        for c in judge_cols:
            lines.append(f"| {c} | {row[c]} |")

        lines += [
            "",
            "**Query:**",
            "",
            f"> {case['query']}",
            "",
            "**Retrieved context (first 500 chars):**",
            "",
            "```text",
            context[:500] + ("…" if len(context) > 500 else ""),
            "```",
            "",
            "**Generated answer:**",
            "",
            "```text",
            answer,
            "```",
            "",
            "---",
            "",
        ]

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports",
            f"judge_disagreement_inspection_{datetime.now().strftime('%Y-%m-%d')}.md",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote disagreement inspection report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path",
                        help="Path to the raw multi-judge CSV from evaluate_generation.py.")
    parser.add_argument("--out", default=None,
                        help="Optional override for the output markdown path.")
    args = parser.parse_args()
    main(args.csv_path, out_path=args.out)
