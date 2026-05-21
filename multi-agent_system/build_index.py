#!/usr/bin/env python3
"""Unified FAISS index builder.

Replaces the previous per-specialty build scripts (`build_cardio_faiss.py`,
`build_endo_faiss.py`). The specialty and its source directory come from
`agents.registry.AGENT_REGISTRY`, so adding a new specialty's index only
requires the registry entry and a populated `data/processed/{specialty}/`
folder — no new build script.

The registry currently exposes four specialists — `cardiologist`,
`endocrinologist`, `gastroenterologist`, `infectionist` — and any of
them can be passed to `--specialty`. Cardiology and endocrinology
indices are pre-built and committed; gastroenterology and infectiology
indices are built on demand with this script.

Usage:
    python build_index.py --specialty cardiologist        # ~40 min, 7,730 chunks
    python build_index.py --specialty endocrinologist     # ~2.5 h,  37,791 chunks
    python build_index.py --specialty gastroenterologist  # ~50 min, 9,024 chunks
    python build_index.py --specialty infectionist        # ~45 min, 7,712 chunks
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


NATIVE_CHUNK_WORDS = 400
NATIVE_OVERLAP_WORDS = 30


# Mean characters-per-whitespace-token above which a chunk is treated as a
# PDF-extraction artifact (inter-word spaces lost during PDF→text conversion,
# so a single "word" is 100+ chars of concatenated text) and skipped before
# embedding. Normal English text averages ~6-7 chars/word; the bloated gastro
# and infection chunks (≈ 5-6 % of those corpora, concentrated in author-
# affiliation blocks of multi-author papers) average 30-170 chars/word and
# trip the Yandex embedder's token-input limit. Threshold = 15 is well above
# normal English and below the artifact range; endocrinology's single 5,696-
# char outlier sits at 14 chars/word and is preserved.
_MAX_MEAN_WORD_LEN_CHARS = 15


def _load_documents(folder_path: str, *, chunk_size: int = NATIVE_CHUNK_WORDS,
                    keep_keywords: bool = False) -> List[Document]:
    """Load chunks for the index build.

    Default behaviour (`chunk_size = NATIVE_CHUNK_WORDS = 400`, `keep_keywords = False`)
    matches the pre-Stage-14 production builder: walk `folder_path` per category,
    strip the `KEYWORDS:` header line from each chunk's text, store keywords in
    metadata only.

    Stage-14 ablation modes:

    * `keep_keywords=True`  — leave the `KEYWORDS:` line *inside* `page_content`
      so the embedder sees the dense token list (cells A and C).
    * `chunk_size != NATIVE_CHUNK_WORDS` — reconstruct the original document text
      by concatenating its native 400-word chunks (in `0001.txt`, `0002.txt`, …
      order, with the 30-word overlap removed), then re-chunk at the requested
      size with overlap `chunk_size // 13` (mirrors the canonical 400/30 ratio).
      Note: raw documents are not on disk in this checkout, so reconstruction
      goes from native chunks rather than the original raw text. Word boundaries
      at the original chunk seams are preserved as-is — see Stage 14 report for
      the methodological caveat.

    Returns a list of `langchain_core.documents.Document` objects with metadata
    {source_file, category, doc_name, keywords}.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size!r}")
    overlap = max(1, chunk_size // 13) if chunk_size != NATIVE_CHUNK_WORDS else NATIVE_OVERLAP_WORDS

    skipped_pdf_artifact = 0  # see _MAX_MEAN_WORD_LEN_CHARS constant

    # First, group source files by (category, doc_name).
    groups: dict = {}
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
                    # JSON chunks (cardiology Articles only — the other three
                    # specialists, endocrinology / gastroenterology / infectiology,
                    # use the conventional 0001.txt … layout for every category)
                    # are kept as a single virtual "document" — no re-chunking
                    # applied (would require text structure we don't have).
                    # Honour keep_keywords for these too: JSON entries do not
                    # carry a KEYWORDS header, so the flag is a no-op here.
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            continue
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    metadata = data.get("metadata", {})
                    groups.setdefault((cat, file_path), [])
                    groups[(cat, file_path)].append({
                        "kind": "json",
                        "text": text,
                        "source_file": metadata.get("source_file") or file,
                        "category": metadata.get("category") or cat,
                        "doc_name": metadata.get("doc_name") or "json",
                        "keywords": metadata.get("keywords", ""),
                    })

                elif file.endswith(".txt") and file.lower() != "summary.txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        continue

                    lines = text.split("\n")
                    body_lines: List[str] = []
                    keyword_line = ""
                    for line in lines:
                        if line.startswith("KEYWORDS:"):
                            keyword_line = line  # full line, including the prefix
                        else:
                            body_lines.append(line)
                    body = "\n".join(body_lines).strip()
                    keywords_meta = keyword_line.replace("KEYWORDS:", "", 1).strip()

                    body_tokens = body.split()
                    if body_tokens:
                        mean_word_chars = sum(len(t) for t in body_tokens) / len(body_tokens)
                        if mean_word_chars > _MAX_MEAN_WORD_LEN_CHARS:
                            skipped_pdf_artifact += 1
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

                    groups.setdefault((category, doc_name), [])
                    groups[(category, doc_name)].append({
                        "kind": "txt",
                        "file": file,
                        "body": body,
                        "keyword_line": keyword_line,
                        "keywords": keywords_meta,
                        "source_file": file,
                        "category": category,
                        "doc_name": doc_name,
                    })

    # Now emit Documents: native chunk-size = native, otherwise re-chunk per doc.
    documents: List[Document] = []
    for key, items in groups.items():
        first = items[0]
        if first["kind"] == "json":
            # JSON: pass through as-is.
            it = items[0]
            documents.append(Document(
                page_content=it["text"],
                metadata={
                    "source_file": it["source_file"],
                    "category": it["category"],
                    "doc_name": it["doc_name"],
                    "keywords": it["keywords"],
                },
            ))
            continue

        if chunk_size == NATIVE_CHUNK_WORDS:
            # Native path: one Document per source .txt chunk.
            for it in items:
                content = it["body"]
                if keep_keywords and it["keyword_line"]:
                    content = it["keyword_line"] + "\n" + content
                if not content.strip():
                    continue
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source_file": it["source_file"],
                        "category": it["category"],
                        "doc_name": it["doc_name"],
                        "keywords": it["keywords"],
                    },
                ))
        else:
            # Re-chunk path: reconstruct body, then re-chunk at the new size.
            sorted_items = sorted(items, key=lambda it: it["file"])
            reconstructed_words: list = []
            for i, it in enumerate(sorted_items):
                w = (it["body"] or "").split()
                if i == 0:
                    reconstructed_words.extend(w)
                else:
                    # Drop the first NATIVE_OVERLAP_WORDS to undo the 400-word sliding window.
                    reconstructed_words.extend(w[NATIVE_OVERLAP_WORDS:])

            # Slide a (chunk_size, overlap) window across reconstructed_words.
            step = chunk_size - overlap
            n = len(reconstructed_words)
            chunk_idx = 0
            start = 0
            keyword_line_first = sorted_items[0]["keyword_line"]
            keywords_meta = sorted_items[0]["keywords"]
            while start < n:
                end = min(start + chunk_size, n)
                body = " ".join(reconstructed_words[start:end])
                if keep_keywords and keyword_line_first:
                    content = keyword_line_first + "\n" + body
                else:
                    content = body
                chunk_idx += 1
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source_file": f"{chunk_idx:04d}.txt",
                        "category": first["category"],
                        "doc_name": first["doc_name"],
                        "keywords": keywords_meta,
                    },
                ))
                if end == n:
                    break
                start += step

    if skipped_pdf_artifact:
        print(
            f"   Skipped {skipped_pdf_artifact} chunk(s) with mean word length > "
            f"{_MAX_MEAN_WORD_LEN_CHARS} chars (PDF-extraction artifacts, "
            f"would exceed the embedder's token-input cap)"
        )
    return documents


def _build_index(specialty: str, *, chunk_size: int = NATIVE_CHUNK_WORDS,
                 keep_keywords: bool = False) -> None:
    if specialty not in AGENT_REGISTRY:
        sys.stderr.write(
            f"Unknown specialty: {specialty!r}. "
            f"Available: {', '.join(sorted(AGENT_REGISTRY))}\n"
        )
        sys.exit(2)

    cfg = AGENT_REGISTRY[specialty]
    folder_path = cfg["folder_path"]

    if chunk_size == NATIVE_CHUNK_WORDS and not keep_keywords:
        # Default cell: write to the registry-canonical folder.
        out_folder = folder_path
    else:
        # Ablation cell: write to a sibling directory so the production index is preserved.
        suffix = f"_{chunk_size}_{'keep' if keep_keywords else 'strip'}"
        out_folder = os.path.join(os.path.dirname(folder_path),
                                  os.path.basename(folder_path) + suffix)
    faiss_save_path = os.path.join(out_folder, "faiss_index")

    print("=" * 60)
    print(f"Building FAISS index for {cfg['name']} Knowledge Base")
    print(f"  Specialty:    {specialty}")
    print(f"  Source dir:   {folder_path}")
    print(f"  Chunk size:   {chunk_size} words "
          f"({'NATIVE' if chunk_size == NATIVE_CHUNK_WORDS else 're-chunked from native'})")
    print(f"  Keywords:     {'KEPT in page_content' if keep_keywords else 'STRIPPED to metadata'}")
    print(f"  Target:       {faiss_save_path}")
    print("=" * 60)

    print("\nLoading documents...")
    documents = _load_documents(folder_path, chunk_size=chunk_size,
                                keep_keywords=keep_keywords)
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
        help=("Specialty key from AGENT_REGISTRY. The registry currently has "
              "four entries: cardiologist, endocrinologist, gastroenterologist, "
              "infectionist."),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=NATIVE_CHUNK_WORDS,
        help=("Words per chunk (default: %(default)s, the native size on disk). "
              "Non-default values trigger re-chunking from the existing native chunks; "
              "output goes to data/processed/{specialty}_{chunk_size}_{keep|strip}/."),
    )
    parser.add_argument(
        "--keep-keywords", action="store_true",
        help=("Leave the `KEYWORDS:` header line inside `page_content` instead of "
              "stripping it to metadata. Used by the Stage 14 ablation."),
    )
    args = parser.parse_args()

    if not YANDEX_API_KEY:
        sys.stderr.write("YANDEX_API_KEY not set. Please set it in .env or environment.\n")
        sys.exit(1)
    if not YANDEX_PROJECT_ID:
        sys.stderr.write("YANDEX_PROJECT_ID not set. Please set it in .env or environment.\n")
        sys.exit(1)

    _build_index(args.specialty, chunk_size=args.chunk_size,
                 keep_keywords=args.keep_keywords)


if __name__ == "__main__":
    main()
