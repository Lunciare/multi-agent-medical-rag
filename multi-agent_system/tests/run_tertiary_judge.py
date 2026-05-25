#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import AGENT_REGISTRY
from judges import JudgeConfig, JudgeStats, judge_faithfulness
from logging_config import configure_logging
from orchestrator import MedicalOrchestrator
from settings import (
    DEFAULT_KNOWLEDGE_BASE_DIR,
    MAX_L2_DISTANCE,
    SIMILARITY_TOP_K,
    TERTIARY_JUDGE_PROVIDER,
)

import judges as _judges_module
import logging

logger = logging.getLogger(__name__)


SPLIT_TO_FILENAME = {
    "dev": "golden_dev.json",
    "test": "golden_test.json",
    "all": "golden_dataset.json",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "reports")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CSV_COLUMNS = ["id", "tier", "domain", "fallback",
               "yandex_primary", "secondary", "tertiary"]


def _load_existing_2judge(csv_path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Existing 2-judge CSV not found at {csv_path}. "
            "Run `evaluate_generation.py --mode multi_judge` once for primary "
            "+ secondary verdicts before invoking this script."
        )
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = r
    return rows


def _load_resume_state(output_csv_path: str) -> set[str]:
    if not os.path.exists(output_csv_path):
        return set()
    done: set[str] = set()
    with open(output_csv_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add(r["id"])
    return done


def _append_row(output_csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLIT_TO_FILENAME), default="test")
    parser.add_argument("--two-judge-csv", default=None,
                        help="Path to the existing 2-judge CSV (default: "
                             "reports/faithfulness_multijudge_raw_2026-05-21.csv)")
    parser.add_argument("--output-csv", default=None,
                        help="Path to the 3-judge output CSV (default: "
                             "reports/faithfulness_multijudge_raw_<today>.csv)")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="OpenRouter retry budget per case; lower = fast-fail (default 1)")
    args = parser.parse_args()

    configure_logging()

    if not TERTIARY_JUDGE_PROVIDER:
        sys.stderr.write("ERROR: TERTIARY_JUDGE_PROVIDER not set in .env.\n")
        sys.exit(2)
    tertiary_cfg = JudgeConfig.from_uri("tertiary", TERTIARY_JUDGE_PROVIDER)
    logger.info("Tertiary judge: provider=%s, model=%s",
                tertiary_cfg.provider, tertiary_cfg.model_id)

    original_max_retries = _judges_module.MAX_RETRIES
    _judges_module.MAX_RETRIES = args.max_retries
    logger.info("OpenRouter MAX_RETRIES temporarily set to %d (was %d).",
                _judges_module.MAX_RETRIES, original_max_retries)

    two_judge_csv = args.two_judge_csv or os.path.join(
        REPORTS_DIR, "faithfulness_multijudge_raw_2026-05-21.csv")
    output_csv = args.output_csv or os.path.join(
        REPORTS_DIR, f"faithfulness_multijudge_raw_{datetime.now().strftime('%Y-%m-%d')}.csv")

    existing = _load_existing_2judge(two_judge_csv)
    logger.info("Loaded %d existing 2-judge rows from %s", len(existing), two_judge_csv)

    already_done = _load_resume_state(output_csv)
    if already_done:
        logger.info("Resuming: %d cases already in %s; will skip them.",
                    len(already_done), output_csv)

    dataset_path = os.path.join(DATA_DIR, SPLIT_TO_FILENAME[args.split])
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    logger.info("Loaded %d cases from %s", len(cases), dataset_path)

    logger.info("Initializing orchestrator + FAISS indices...")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)

    stats = JudgeStats()
    n_done = n_fallback = n_tertiary_call = n_tertiary_none = 0
    start_t = time.time()

    for case in cases:
        cid = case["id"]
        if cid in already_done:
            n_done += 1
            continue

        prior = existing.get(cid)
        if prior is None:
            logger.warning("Case %s not in 2-judge CSV; skipping.", cid)
            continue

        expected_agent = case["expected_specialist"]
        agent = orchestrator.agents.get(expected_agent)
        if agent is None:
            logger.warning("No agent registered for %s; skipping %s.", expected_agent, cid)
            continue

        row = {
            "id": cid,
            "tier": case.get("tier", 1),
            "domain": expected_agent,
            "fallback": prior["fallback"],
            "yandex_primary": prior["yandex_primary"],
            "secondary": prior["secondary"],
            "tertiary": "",
        }

        if prior["fallback"].lower() == "true":
            row["tertiary"] = "FALLBACK"
            _append_row(output_csv, row)
            n_fallback += 1
            logger.info("[%s] FALLBACK (skipped per 2-judge CSV)", cid)
            continue

        query = case["query"]
        docs_and_scores = agent.vectorstore.similarity_search_with_score(
            query, k=SIMILARITY_TOP_K)
        docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
        context = "\n\n".join(doc.page_content for doc in docs)
        _spec, answer, _ev = agent.answer(query)

        verdict = judge_faithfulness(query, context, answer, tertiary_cfg,
                                     case_id=cid, stats=stats)
        if verdict is True:
            row["tertiary"] = "FAITHFUL"
        elif verdict is False:
            row["tertiary"] = "HALLUCINATION"
        else:
            row["tertiary"] = "NONE"
            n_tertiary_none += 1
        n_tertiary_call += 1
        _append_row(output_csv, row)
        logger.info("[%s] tertiary=%s  (cumulative: %d done, %d NONE)",
                    cid, row["tertiary"], n_tertiary_call, n_tertiary_none)

    elapsed = time.time() - start_t
    logger.info("Finished. Wall-clock %.1fs (%.1f min). "
                "tertiary calls=%d, NONE=%d, FALLBACK=%d, already-done=%d. "
                "JudgeStats: successes=%d, http_errors=%d, exhausted=%d.",
                elapsed, elapsed / 60, n_tertiary_call, n_tertiary_none,
                n_fallback, n_done, stats.successes, stats.http_errors, stats.exhausted)
    logger.info("Output CSV: %s", output_csv)


if __name__ == "__main__":
    main()
