#!/usr/bin/env python3
"""Unified FAISS index builder.

Replaces the previous per-specialty build scripts (`build_cardio_faiss.py`,
`build_endo_faiss.py`). The specialty and its source directory come from
`agents.registry.AGENT_REGISTRY`, so adding a new specialty's index only
requires the registry entry and a populated `data/processed/{specialty}/`
folder — no new build script.

Usage:
    python build_index.py --specialty cardiologist
    python build_index.py --specialty endocrinologist
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from agents.registry import AGENT_REGISTRY
from embeddings import YandexNativeEmbeddings
from settings import YANDEX_API_KEY, YANDEX_PROJECT_ID


CATEGORIES = ["Articles", "Cases", "Guidelines", "Handbooks", "Textbooks"]
SAVE_EVERY_N = 500
MAX_WORKERS = 3


def _load_documents(folder_path: str) -> List[Document]:
    """Mirror SpecialistAgent._load_documents but with per-category walking
    + doc_name metadata that the original build scripts emitted.
    """
    documents: List[Document] = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(folder_path, cat)
        if not os.path.isdir(cat_dir):
            continue

        for root, _dirs, files in os.walk(cat_dir):
            if "faiss_index" in root:
                continue

            for file in sorted(files):
                file_path = os.path.join(root, file)

                if file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            continue
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    metadata = data.get("metadata", {})
                    documents.append(
                        Document(page_content=text, metadata=metadata)
                    )

                elif file.endswith(".txt") and file.lower() != "summary.txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        continue

                    lines = text.split("\n")
                    clean_lines = []
                    keywords = ""
                    for line in lines:
                        if line.startswith("KEYWORDS:"):
                            keywords = line.replace("KEYWORDS:", "").strip()
                        else:
                            clean_lines.append(line)
                    clean_text = "\n".join(clean_lines).strip()
                    if not clean_text:
                        continue

                    parts = file_path.replace("\\", "/").split("/")
                    category = "unknown"
                    doc_name = "unknown"
                    for i, p in enumerate(parts):
                        if p in CATEGORIES:
                            category = p
                            if i + 1 < len(parts) - 1:
                                doc_name = parts[i + 1]
                            break

                    documents.append(
                        Document(
                            page_content=clean_text,
                            metadata={
                                "source_file": file,
                                "category": category,
                                "doc_name": doc_name,
                                "keywords": keywords,
                            },
                        )
                    )
    return documents


def _build_index(specialty: str) -> None:
    if specialty not in AGENT_REGISTRY:
        sys.stderr.write(
            f"Unknown specialty: {specialty!r}. "
            f"Available: {', '.join(sorted(AGENT_REGISTRY))}\n"
        )
        sys.exit(2)

    cfg = AGENT_REGISTRY[specialty]
    folder_path = cfg["folder_path"]
    faiss_save_path = os.path.join(folder_path, "faiss_index")

    print("=" * 60)
    print(f"Building FAISS index for {cfg['name']} Knowledge Base")
    print(f"Source: {folder_path}")
    print(f"Target: {faiss_save_path}")
    print("=" * 60)

    print("\nLoading documents...")
    documents = _load_documents(folder_path)
    print(f"   Found {len(documents)} chunks across all categories")
    if not documents:
        print("No documents found!")
        sys.exit(1)

    cat_counts = Counter(d.metadata.get("category", "unknown") for d in documents)
    for cat, count in sorted(cat_counts.items()):
        print(f"   {cat}: {count} chunks")

    print("\nInitializing Yandex embeddings...")
    embeddings = YandexNativeEmbeddings()

    print(f"\nBuilding FAISS index ({len(documents)} chunks)...")
    print(f"   Workers: {MAX_WORKERS} parallel | Batch size: {SAVE_EVERY_N}")
    print(f"   Estimated time: ~{len(documents) / MAX_WORKERS / 5 / 60:.1f} hours")

    os.makedirs(faiss_save_path, exist_ok=True)

    vectorstore = None
    start_idx = 0
    index_file = os.path.join(faiss_save_path, "index.faiss")
    if os.path.exists(index_file):
        print(f"\nFound existing index at {faiss_save_path}, loading to resume...")
        vectorstore = FAISS.load_local(
            faiss_save_path, embeddings, allow_dangerous_deserialization=True
        )
        start_idx = vectorstore.index.ntotal
        print(f"Resuming from chunk {start_idx}...")

    for start in range(start_idx, len(documents), SAVE_EVERY_N):
        end = min(start + SAVE_EVERY_N, len(documents))
        batch = documents[start:end]

        batch_num = start // SAVE_EVERY_N + 1
        total_batches = (len(documents) + SAVE_EVERY_N - 1) // SAVE_EVERY_N
        print(f"\nBatch {batch_num}/{total_batches}: chunks {start+1}-{end}")

        t0 = time.time()
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
        elapsed = time.time() - t0
        rate = len(batch) / elapsed if elapsed > 0 else 0
        remaining = len(documents) - end
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"Done in {elapsed:.1f}s ({rate:.1f} chunks/s) — ETA: {eta:.1f} min remaining")

        print(f"Saving progress to {faiss_save_path}...")
        vectorstore.save_local(faiss_save_path)

    print(f"\n{'='*60}")
    print(f"FAISS index built and saved successfully!")
    print(f"Location: {faiss_save_path}")
    print(f"Total chunks indexed: {len(documents)}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified FAISS index builder. Reads specialty config from "
                    "agents/registry.py and builds the index for one specialty.")
    parser.add_argument(
        "--specialty", required=True,
        choices=sorted(AGENT_REGISTRY.keys()),
        help="Specialty key from AGENT_REGISTRY (e.g. cardiologist, endocrinologist).",
    )
    args = parser.parse_args()

    if not YANDEX_API_KEY:
        sys.stderr.write("YANDEX_API_KEY not set. Please set it in .env or environment.\n")
        sys.exit(1)
    if not YANDEX_PROJECT_ID:
        sys.stderr.write("YANDEX_PROJECT_ID not set. Please set it in .env or environment.\n")
        sys.exit(1)

    _build_index(args.specialty)


if __name__ == "__main__":
    main()
