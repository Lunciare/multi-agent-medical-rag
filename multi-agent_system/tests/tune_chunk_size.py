#!/usr/bin/env python3
import argparse
import os
import sys
import json
from collections import defaultdict
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embeddings import YandexNativeEmbeddings
from settings import (
    MAX_L2_DISTANCE, SIMILARITY_TOP_K,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARDIO_DIR  = os.path.join(BASE_DIR, "data", "processed", "cardiology")
ENDO_DIR    = os.path.join(BASE_DIR, "data", "processed", "endocrinology")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_dev.json")

CHUNK_SIZES_TO_TEST = [100, 200, 400, 500, 600]
CHUNK_OVERLAP_WORDS = 30

MAX_DOCS_PER_SPECIALTY = 10


def words(text: str) -> List[str]:
    return text.split()

def rechunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    ws = words(text)
    if not ws:
        return []
    chunks = []
    start = 0
    while start < len(ws):
        end = min(start + chunk_size, len(ws))
        chunks.append(" ".join(ws[start:end]))
        if end == len(ws):
            break
        start += chunk_size - overlap
    return chunks

def strip_keywords(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n")
        if not line.startswith("KEYWORDS:")
    ).strip()

def load_chunks_from_dir(data_dir: str) -> Dict[str, List[str]]:
    doc_chunks: Dict[str, List[str]] = defaultdict(list)
    for root, dirs, files in os.walk(data_dir):
        if "faiss_index" in root:
            continue
        for file in sorted(files):
            if not file.endswith(".txt") or file.lower() == "summary.txt":
                continue
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                text = strip_keywords(f.read().strip())
            if text:
                doc_name = os.path.basename(root)
                doc_chunks[doc_name].append(text)
    return doc_chunks


def main():
    print("=" * 65)
    print("Chunk Size Grid Search")
    print("=" * 65)

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    cardio_cases = [c for c in dataset if c["expected_specialist"] == "cardiologist"]
    endo_cases   = [c for c in dataset if c["expected_specialist"] == "endocrinologist"]

    def case_keywords(case):
        return [kw.lower() for kw in case.get("expected_keywords", [])]

    cardio_keywords_all = {kw for c in cardio_cases for kw in case_keywords(c)}
    endo_keywords_all   = {kw for c in endo_cases   for kw in case_keywords(c)}

    print(f"\nLoading document chunks...")
    cardio_docs = load_chunks_from_dir(CARDIO_DIR)
    endo_docs   = load_chunks_from_dir(ENDO_DIR)
    print(f"   Cardiology: {len(cardio_docs)} source documents")
    print(f"   Endocrinology: {len(endo_docs)} source documents")

    def select_relevant(doc_chunks: Dict[str, List[str]], keywords: set, max_docs: int):
        selected = {}
        for doc_name, chunks in doc_chunks.items():
            full_text = " ".join(chunks).lower()
            if any(kw in full_text for kw in keywords):
                selected[doc_name] = " ".join(chunks)
            if len(selected) >= max_docs:
                break
        return selected

    cardio_relevant = select_relevant(cardio_docs, cardio_keywords_all, MAX_DOCS_PER_SPECIALTY)
    endo_relevant   = select_relevant(endo_docs,   endo_keywords_all,   MAX_DOCS_PER_SPECIALTY)
    print(f"\n🔍 Relevant document subset selected:")
    print(f"   Cardiology: {len(cardio_relevant)} docs")
    print(f"   Endocrinology: {len(endo_relevant)} docs")

    embeddings = YandexNativeEmbeddings()

    results = []

    print(f"\n{'─'*65}")
    print(f"{'Chunk Size':<12} {'Docs Indexed':<14} {'Hit Rate':<12} {'Avg Retrieved'}")
    print(f"{'─'*65}")

    for chunk_size in CHUNK_SIZES_TO_TEST:
        cardio_rechunked: List[Document] = []
        for doc_name, full_text in cardio_relevant.items():
            for chunk_text in rechunk(full_text, chunk_size, CHUNK_OVERLAP_WORDS):
                cardio_rechunked.append(Document(
                    page_content=chunk_text,
                    metadata={"source": doc_name, "specialty": "cardiology"},
                ))

        endo_rechunked: List[Document] = []
        for doc_name, full_text in endo_relevant.items():
            for chunk_text in rechunk(full_text, chunk_size, CHUNK_OVERLAP_WORDS):
                endo_rechunked.append(Document(
                    page_content=chunk_text,
                    metadata={"source": doc_name, "specialty": "endocrinology"},
                ))

        total_chunks = len(cardio_rechunked) + len(endo_rechunked)

        try:
            cardio_vs = FAISS.from_documents(cardio_rechunked, embeddings) if cardio_rechunked else None
            endo_vs   = FAISS.from_documents(endo_rechunked,   embeddings) if endo_rechunked   else None
        except Exception as e:
            print(f"FAISS build failed for chunk_size={chunk_size}: {e}")
            continue

        hits = 0
        total_retrieved = 0

        for case in dataset:
            query    = case["query"]
            spec     = case["expected_specialist"]
            keywords = case_keywords(case)

            vs = cardio_vs if spec == "cardiologist" else endo_vs
            if vs is None:
                continue

            docs_and_scores = vs.similarity_search_with_score(query, k=SIMILARITY_TOP_K)
            docs = [doc for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]
            total_retrieved += len(docs)

            retrieved_text = " ".join(d.page_content.lower() for d in docs)
            if any(kw in retrieved_text for kw in keywords):
                hits += 1

        n = len(dataset)
        hit_rate = hits / n * 100 if n > 0 else 0
        avg_retrieved = total_retrieved / n if n > 0 else 0
        results.append((chunk_size, hit_rate, avg_retrieved, total_chunks))

        print(f"{chunk_size:<12} {total_chunks:<14} {hit_rate:<11.1f}% {avg_retrieved:.1f}")

    print(f"{'─'*65}")
    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\nBest chunk size: {best[0]} words "
              f"(Hit Rate: {best[1]:.1f}%, Avg retrieved: {best[2]:.1f})")
        print(f"\nNote: this evaluation uses a subset of ~{MAX_DOCS_PER_SPECIALTY} relevant docs")
        print(f"   per specialty. Run a full rebuild with the winning chunk size to confirm.")

    print(f"\n{'='*65}")
    print("Grid search complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="dev",
        help="Tuning is locked to the dev split (golden_dev.json). "
             "Any value other than 'dev' is rejected.",
    )
    args = parser.parse_args()
    if args.split != "dev":
        sys.stderr.write(
            f"tune_chunk_size.py is locked to --split dev "
            f"(got --split {args.split}). Hyperparameters must be tuned on the "
            f"dev split only; evaluate on test via evaluate_*.py --split test.\n"
        )
        sys.exit(2)
    main()
