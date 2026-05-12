#!/usr/bin/env python3
import os
import re
import sys
import time
import json
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embeddings import YandexNativeEmbeddings
from settings import (
    YANDEX_API_KEY,
    YANDEX_PROJECT_ID,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_MODEL,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDO_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "endocrinology")
FAISS_SAVE_PATH = os.path.join(ENDO_DATA_DIR, "faiss_index")

CATEGORIES = ["Articles", "Cases", "Guidelines", "Handbooks", "Textbooks"]

MAX_WORKERS = 3
SAVE_EVERY_N = 500


def load_endocrinology_documents() -> List[Document]:
    documents = []

    for cat in CATEGORIES:
        cat_dir = os.path.join(ENDO_DATA_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue

        for root, dirs, files in os.walk(cat_dir):
            if "faiss_index" in root:
                continue

            for file in sorted(files):
                if not file.endswith(".txt"):
                    continue
                if file.lower() == "summary.txt":
                    continue

                file_path = os.path.join(root, file)
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


def main():
    if not YANDEX_API_KEY:
        print("YANDEX_API_KEY not set. Please set it in .env or environment.")
        sys.exit(1)
    if not YANDEX_PROJECT_ID:
        print("YANDEX_PROJECT_ID not set. Please set it in .env or environment.")
        sys.exit(1)

    print("=" * 60)
    print("Building FAISS index for Endocrinology Knowledge Base")
    print("=" * 60)

    print("\nLoading documents...")
    documents = load_endocrinology_documents()
    print(f"Found {len(documents)} chunks across all categories")

    if not documents:
        print("No documents found!")
        sys.exit(1)

    from collections import Counter
    cat_counts = Counter(d.metadata["category"] for d in documents)
    for cat, count in sorted(cat_counts.items()):
        print(f"   {cat}: {count} chunks")

    print("\nInitializing Yandex embeddings...")
    embeddings = YandexNativeEmbeddings()

    print(f"\nBuilding FAISS index ({len(documents)} chunks)...")
    print(f"   This will take a while with {MAX_WORKERS} parallel workers...")
    print(f"   Estimated time: ~{len(documents) / MAX_WORKERS / 5:.0f} minutes")

    os.makedirs(FAISS_SAVE_PATH, exist_ok=True)

    batch_size = SAVE_EVERY_N
    vectorstore = None
    start_idx = 0

    index_file = os.path.join(FAISS_SAVE_PATH, "index.faiss")
    if os.path.exists(index_file):
        print(f"\nFound existing index at {FAISS_SAVE_PATH}, loading to resume...")
        vectorstore = FAISS.load_local(FAISS_SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
        start_idx = vectorstore.index.ntotal
        print(f"Resuming from chunk {start_idx}...")

    for start in range(start_idx, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        batch = documents[start:end]

        batch_num = start // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
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

        print(f"Saving progress to {FAISS_SAVE_PATH}...")
        vectorstore.save_local(FAISS_SAVE_PATH)

    print(f"\n{'='*60}")
    print(f"FAISS index built and saved successfully!")
    print(f"Location: {FAISS_SAVE_PATH}")
    print(f"Total chunks indexed: {len(documents)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
