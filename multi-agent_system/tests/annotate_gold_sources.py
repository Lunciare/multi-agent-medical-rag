#!/usr/bin/env python3
"""Annotate `gold_sources` on golden_dataset.json.

Two modes:

  --interactive : for each Tier 1/2 case, prints the top-K retrieved sources and
                  prompts [y/n/s/q] per document. Pick up to N gold docs per case.
                  Marks every Tier 3 case as gold_sources=[].
                  This is the workflow described in the Stage 6 task.

  --auto        : non-interactive heuristic. For each Tier 1/2 case, retrieves
                  the top-K chunks, aggregates by doc_name, ranks by (unique
                  keyword hits desc, first appearance rank asc, chunk count desc),
                  and writes the top documents that have ≥1 expected-keyword hit
                  back as gold_sources. Tier 3 cases get [].
                  Used as the initial pass so the metric is computable without
                  manual labour; a student can refine via --interactive later.

Both modes write back to multi-agent_system/tests/data/golden_dataset.json
(and propagate the changes to golden_dev.json + golden_test.json).
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR, MAX_L2_DISTANCE


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GOLDEN_PATH = os.path.join(DATA_DIR, "golden_dataset.json")
DEV_PATH = os.path.join(DATA_DIR, "golden_dev.json")
TEST_PATH = os.path.join(DATA_DIR, "golden_test.json")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path, dataset):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _retrieve_top_k(orchestrator, case, top_k):
    agent = (orchestrator.cardiologist if case["expected_specialist"] == "cardiologist"
             else orchestrator.endocrinologist)
    return agent.vectorstore.similarity_search_with_score(case["query"], k=top_k)


def _aggregate_docs(docs_and_scores, expected_keywords):
    """Group retrieved chunks by doc_name. Returns a list of dicts ranked by
    (unique_kw_hits desc, first_rank asc, chunks desc).
    """
    kw_lower = [k.lower() for k in expected_keywords]
    by_doc = {}
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        md = doc.metadata or {}
        dn = md.get("doc_name") or md.get("source_file") or "?"
        text = (doc.page_content or "").lower()
        hits = {kw for kw in kw_lower if kw in text}
        if dn not in by_doc:
            by_doc[dn] = {
                "doc_name": dn,
                "source_file": md.get("source_file", "?"),
                "category": md.get("category", "?"),
                "first_rank": rank,
                "first_l2": score,
                "chunks": 0,
                "kw_hits": set(),
                "within_threshold": score <= MAX_L2_DISTANCE,
            }
        by_doc[dn]["chunks"] += 1
        by_doc[dn]["kw_hits"].update(hits)
        if score <= MAX_L2_DISTANCE:
            by_doc[dn]["within_threshold"] = True

    ranked = sorted(
        by_doc.values(),
        key=lambda d: (-len(d["kw_hits"]), d["first_rank"], -d["chunks"]),
    )
    return ranked


def _gold_entry(doc):
    return {"source_file": doc["source_file"], "doc_name": doc["doc_name"]}


def auto_annotate(dataset, orchestrator, top_k, max_gold):
    annotated = 0
    empty = 0
    for case in dataset:
        tier = case.get("tier", 1)
        if tier == 3:
            case["gold_sources"] = []
            continue

        ds = _retrieve_top_k(orchestrator, case, top_k)
        ranked = _aggregate_docs(ds, case["expected_keywords"])
        picked = [d for d in ranked if d["kw_hits"] and d["within_threshold"]][:max_gold]
        case["gold_sources"] = [_gold_entry(d) for d in picked]
        if case["gold_sources"]:
            annotated += 1
            print(f"  {case['id']:<12} [auto] tier={tier}: "
                  f"picked {len(case['gold_sources'])} doc(s) — "
                  + ", ".join(d["doc_name"][:40] for d in picked))
        else:
            empty += 1
            print(f"  {case['id']:<12} [auto] tier={tier}: NO match (gold_sources=[])")
    return annotated, empty


def interactive_annotate(dataset, orchestrator, top_k, max_gold):
    print("\nInteractive annotation. For each case, mark each shown doc:")
    print("  y = include as gold source")
    print("  n = skip (default if you just press Enter)")
    print("  s = skip rest of this case")
    print("  q = quit and save\n")

    annotated = 0
    for case in dataset:
        tier = case.get("tier", 1)
        if tier == 3:
            case["gold_sources"] = []
            continue

        print("=" * 80)
        print(f"{case['id']}  tier={tier}/{case['tier_label']}  "
              f"domain={case['expected_specialist']}")
        print(f"Query: {case['query']}")
        print(f"Expected keywords: {', '.join(case['expected_keywords'])}")
        existing = case.get("gold_sources")
        if existing:
            print(f"Existing gold_sources: "
                  + ", ".join(g.get("doc_name", g.get("source_file", "?"))[:40]
                              for g in existing))
        print()

        ds = _retrieve_top_k(orchestrator, case, top_k)
        ranked = _aggregate_docs(ds, case["expected_keywords"])
        picked = []
        for d in ranked:
            if len(picked) >= max_gold:
                break
            tag = "" if d["within_threshold"] else "  (BEYOND L2_THRESHOLD)"
            print(f"  doc_name : {d['doc_name']}{tag}")
            print(f"  category : {d['category']}  first_rank={d['first_rank']}  "
                  f"chunks={d['chunks']}  kw hits={len(d['kw_hits'])}/{len(case['expected_keywords'])}"
                  f"  ({', '.join(sorted(d['kw_hits']))})")
            ans = input("  include? [y/N/s/q] ").strip().lower()
            if ans == "q":
                case["gold_sources"] = picked
                if picked:
                    annotated += 1
                return annotated, "quit"
            if ans == "s":
                break
            if ans == "y":
                picked.append(_gold_entry(d))
            print()
        case["gold_sources"] = picked
        if picked:
            annotated += 1
    return annotated, "done"


def propagate_splits(dataset):
    by_id = {c["id"]: c for c in dataset}
    for path in (DEV_PATH, TEST_PATH):
        if not os.path.exists(path):
            continue
        sub = _load(path)
        for c in sub:
            updated = by_id.get(c["id"])
            if updated is not None:
                c["gold_sources"] = updated.get("gold_sources", [])
        _save(path, sub)
        print(f"Updated {path} ({len(sub)} cases).")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto", action="store_true",
                      help="Programmatically pick gold sources via the keyword "
                           "coverage heuristic. Non-interactive.")
    mode.add_argument("--interactive", action="store_true",
                      help="Prompt y/n per document, per Tier 1/2 case.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K to retrieve per case (default 20).")
    parser.add_argument("--max-gold", type=int, default=3,
                        help="Max gold documents per case (default 3).")
    args = parser.parse_args()

    print(f"Loading golden dataset from {GOLDEN_PATH}...")
    dataset = _load(GOLDEN_PATH)
    print(f"Loaded {len(dataset)} cases.")
    print("Loading FAISS indices (this may take a moment)...")
    orchestrator = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)

    if args.auto:
        annotated, empty = auto_annotate(dataset, orchestrator,
                                         top_k=args.top_k, max_gold=args.max_gold)
        print(f"\nAuto-annotation done: {annotated} Tier1/2 cases got >=1 gold source; "
              f"{empty} got no match.")
    else:
        annotated, status = interactive_annotate(dataset, orchestrator,
                                                 top_k=args.top_k,
                                                 max_gold=args.max_gold)
        print(f"\nInteractive annotation ended ({status}); "
              f"{annotated} Tier1/2 cases now have gold_sources.")

    _save(GOLDEN_PATH, dataset)
    print(f"Saved {GOLDEN_PATH}.")
    propagate_splits(dataset)


if __name__ == "__main__":
    main()
