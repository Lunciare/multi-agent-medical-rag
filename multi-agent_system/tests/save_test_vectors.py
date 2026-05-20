"""Regenerate the retrieval regression snapshot.

Produces three on-disk artefacts under `multi-agent_system/tests/data/`:
  - `test_vectors.npy`         — Yandex query embeddings for the 10 canonical
                                  regression queries (5 cardiology + 5 endo).
  - `test_vector_labels.json`  — parallel list of {query, domain}.
  - `test_retrieval_snapshot.json` — for each canonical query: the top-K=5
    `source_file`s returned by the corresponding FAISS index and the matching
    L2 distances. Stage 16 regression test (`tests/test_retrieval_regression.py`)
    diffs the live retrieval against this snapshot.

Usage:
    cd multi-agent_system
    python tests/save_test_vectors.py                 # build .npy + .json only IF MISSING
    python tests/save_test_vectors.py --update-snapshot  # also (re)write retrieval snapshot

The `--update-snapshot` flag is intentional: it forces a rewrite of
`test_retrieval_snapshot.json` after a legitimate index rebuild (chunking,
embeddings, K, or L2-threshold change). Without the flag, the script refuses
to overwrite an existing snapshot.

Requires YANDEX_API_KEY / YANDEX_PROJECT_ID (loaded from `multi-agent_system/.env`).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings import YandexNativeEmbeddings


CARDIOLOGY_QUERIES = [
    "atrial fibrillation",
    "hypertension management",
    "chest pain differential",
    "echocardiography",
    "beta blocker",
]
ENDOCRINOLOGY_QUERIES = [
    "type 2 diabetes",
    "thyroid nodule",
    "insulin resistance",
    "HbA1c",
    "adrenal insufficiency",
]


SIMILARITY_TOP_K = 5


def _faiss_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "cardiology":    repo_root / "data" / "processed" / "cardiology"    / "faiss_index",
        "endocrinology": repo_root / "data" / "processed" / "endocrinology" / "faiss_index",
    }


def _load_faiss(domain: str, faiss_dir: Path):
    """Load a LangChain FAISS vectorstore for `domain` from `faiss_dir`."""
    from langchain_community.vectorstores import FAISS
    embedder = YandexNativeEmbeddings()
    return FAISS.load_local(str(faiss_dir), embedder, allow_dangerous_deserialization=True)


def _query_snapshot(vs, vec: np.ndarray, k: int = SIMILARITY_TOP_K) -> dict:
    """Run a single embedded query against `vs` and return the top-K snapshot.

    Uses the vectorstore's raw `similarity_search_with_score_by_vector` so the
    snapshot reflects exactly what `Agent.answer` would see at runtime (same
    L2 distance, same source_file ordering).
    """
    # `similarity_search_with_score_by_vector` returns [(Document, distance), ...].
    pairs = vs.similarity_search_with_score_by_vector(vec.tolist(), k=k)
    source_files = []
    l2_distances = []
    for doc, dist in pairs:
        md = doc.metadata or {}
        sf = md.get("source_file") or md.get("doc_name") or "?"
        source_files.append(sf)
        l2_distances.append(float(dist))
    return {"top_k_source_files": source_files, "top_k_l2_distances": l2_distances}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-snapshot", action="store_true",
        help=("Overwrite test_retrieval_snapshot.json. Use this after a "
              "deliberate index rebuild (Fix 11 / Stage 14-style chunking or "
              "embedding change). Without this flag, an existing snapshot is "
              "preserved and only the vectors/labels are refreshed."),
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent.parent

    vectors_path = out_dir / "test_vectors.npy"
    labels_path  = out_dir / "test_vector_labels.json"
    snapshot_path = out_dir / "test_retrieval_snapshot.json"

    embedder = YandexNativeEmbeddings()
    pairs = (
        [(q, "cardiology")    for q in CARDIOLOGY_QUERIES]
        + [(q, "endocrinology") for q in ENDOCRINOLOGY_QUERIES]
    )

    vectors = []
    labels = []
    for query, domain in pairs:
        print(f"Embedding [{domain}] '{query}'…")
        vectors.append(embedder.embed_query(query))
        labels.append({"query": query, "domain": domain})

    arr = np.asarray(vectors, dtype=np.float32)
    np.save(vectors_path, arr)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"\nSaved {arr.shape} vectors → {vectors_path}")
    print(f"Saved {len(labels)} labels  → {labels_path}")

    # ----- snapshot stage -----
    if snapshot_path.exists() and not args.update_snapshot:
        print(f"\nSnapshot already exists at {snapshot_path}; not overwriting. "
              f"Re-run with --update-snapshot to refresh it after a legitimate "
              f"index rebuild.")
        return

    faiss_paths = _faiss_paths(repo_root)
    vectorstores: dict[str, object] = {}
    for domain, path in faiss_paths.items():
        if not (path / "index.faiss").exists():
            print(f"WARNING: {domain} FAISS index missing at {path}; "
                  f"snapshot will skip {domain} queries.")
            continue
        print(f"Loading {domain} FAISS index from {path}…")
        vectorstores[domain] = _load_faiss(domain, path)

    snapshot: dict = {}
    for vec, label in zip(arr, labels):
        domain = label["domain"]
        query = label["query"]
        if domain not in vectorstores:
            continue
        entry = _query_snapshot(vectorstores[domain], vec, k=SIMILARITY_TOP_K)
        entry["domain"] = domain
        entry["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
        # Keep insertion order: query → entry.
        snapshot[query] = entry
        files = entry["top_k_source_files"]
        dists = entry["top_k_l2_distances"]
        print(f"  [{domain}] {query!r} → top-{len(files)} sources, "
              f"L2 ∈ [{min(dists):.3f}, {max(dists):.3f}]")

    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"\nSaved {len(snapshot)}-query retrieval snapshot → {snapshot_path}")


if __name__ == "__main__":
    main()
