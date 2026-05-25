#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS

from agents.registry import AGENT_REGISTRY
from embeddings import YandexNativeEmbeddings


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 2]


def _load_chunks_from_faiss(faiss_index_path: str) -> Tuple[List[str], List[dict]]:
    embeddings = YandexNativeEmbeddings()
    print(f"  Loading FAISS docstore from {faiss_index_path}…")
    vs = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    docs = list(vs.docstore._dict.values())
    print(f"  Found {len(docs)} chunks in the cached FAISS index")
    texts = [d.page_content for d in docs]
    metas = [dict(d.metadata) for d in docs]
    return texts, metas


def build_one(specialty_key: str) -> dict:
    if specialty_key not in AGENT_REGISTRY:
        sys.stderr.write(
            f"Unknown specialty: {specialty_key!r}. "
            f"Available: {', '.join(sorted(AGENT_REGISTRY))}\n"
        )
        sys.exit(2)

    cfg = AGENT_REGISTRY[specialty_key]
    folder_path = cfg["folder_path"]
    faiss_index_path = os.path.join(folder_path, "faiss_index")
    bm25_pickle_path = os.path.join(folder_path, "bm25_index.pkl")

    print("=" * 60)
    print(f"Building BM25 index for {cfg['name']}")
    print(f"  Source FAISS: {faiss_index_path}")
    print(f"  Target pickle: {bm25_pickle_path}")
    print("=" * 60)

    texts, metas = _load_chunks_from_faiss(faiss_index_path)

    print(f"  Tokenising {len(texts)} chunks…")
    t0 = time.time()
    tokens = [tokenize(t) for t in texts]
    tokenise_s = time.time() - t0
    print(f"  Tokenisation took {tokenise_s:.1f}s "
          f"(avg {sum(len(toks) for toks in tokens)/max(1, len(tokens)):.1f} tokens/chunk)")

    from rank_bm25 import BM25Okapi
    t0 = time.time()
    bm25 = BM25Okapi(tokens)
    build_s = time.time() - t0
    print(f"  BM25Okapi() build took {build_s:.1f}s")

    payload = {
        "specialty": specialty_key,
        "n_chunks": len(texts),
        "bm25": bm25,
        "metadatas": metas,
    }
    t0 = time.time()
    with open(bm25_pickle_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_s = time.time() - t0
    pickle_bytes = os.path.getsize(bm25_pickle_path)
    print(f"  Pickle write took {pickle_s:.1f}s; on-disk size = "
          f"{pickle_bytes/1024/1024:.1f} MiB ({pickle_bytes:,} bytes)")
    print("=" * 60)

    return {
        "specialty": specialty_key,
        "n_chunks": len(texts),
        "tokenise_s": tokenise_s,
        "build_s": build_s,
        "pickle_s": pickle_s,
        "pickle_bytes": pickle_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specialty", required=True,
        choices=sorted(AGENT_REGISTRY.keys()) + ["all"],
        help="Specialty key from AGENT_REGISTRY, or 'all' to build both.",
    )
    args = parser.parse_args()

    targets = sorted(AGENT_REGISTRY.keys()) if args.specialty == "all" else [args.specialty]
    summaries = []
    for sp in targets:
        summaries.append(build_one(sp))

    print("\nSummary:")
    print(f"  {'specialty':<18} {'chunks':>8} {'tokenise':>10} {'build':>8} "
          f"{'pickle MiB':>12}")
    for s in summaries:
        print(f"  {s['specialty']:<18} {s['n_chunks']:>8} "
              f"{s['tokenise_s']:>9.1f}s {s['build_s']:>7.1f}s "
              f"{s['pickle_bytes']/1024/1024:>11.1f}")


if __name__ == "__main__":
    main()
