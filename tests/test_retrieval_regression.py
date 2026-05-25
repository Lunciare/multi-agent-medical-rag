
import json
from pathlib import Path

import faiss
import numpy as np
import pytest


SIMILARITY_TOP_K = 5
L2_DRIFT_TOLERANCE = 0.1

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "multi-agent_system" / "tests" / "data"
VECTORS_PATH = TEST_DATA_DIR / "test_vectors.npy"
LABELS_PATH = TEST_DATA_DIR / "test_vector_labels.json"
SNAPSHOT_PATH = TEST_DATA_DIR / "test_retrieval_snapshot.json"

FAISS_DIRS = {
    "cardiology":         REPO_ROOT / "data" / "processed" / "cardiology"         / "faiss_index",
    "endocrinology":      REPO_ROOT / "data" / "processed" / "endocrinology"      / "faiss_index",
    "gastroenterologist": REPO_ROOT / "data" / "processed" / "gastroenterologist" / "faiss_index",
    "infection":          REPO_ROOT / "data" / "processed" / "infection"          / "faiss_index",
}


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        pytest.skip(f"{hint} not found at {path} — run save_test_vectors.py first.")


@pytest.fixture(scope="module")
def saved_vectors():
    _require(VECTORS_PATH, "test_vectors.npy")
    _require(LABELS_PATH, "test_vector_labels.json")
    vectors = np.load(VECTORS_PATH).astype(np.float32)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    assert vectors.shape[0] == len(labels), "vectors/labels length mismatch"
    return vectors, labels


@pytest.fixture(scope="module")
def snapshot():
    _require(SNAPSHOT_PATH, "test_retrieval_snapshot.json")
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vectorstores():
    import pickle, sys as _sys

    for _m in list(_sys.modules):
        if _m.startswith("langchain_core") or _m.startswith("langchain_community"):
            _sys.modules.pop(_m, None)

    stores = {}
    for domain, faiss_dir in FAISS_DIRS.items():
        _require(faiss_dir / "index.faiss", f"{domain} FAISS index")
        _require(faiss_dir / "index.pkl",   f"{domain} FAISS docstore pickle")
        idx = faiss.read_index(str(faiss_dir / "index.faiss"))
        with open(faiss_dir / "index.pkl", "rb") as f:
            docstore, idx_to_id = pickle.load(f)
        stores[domain] = {"index": idx, "docstore": docstore, "idx_to_id": idx_to_id}
    return stores


def _retrieve_topk(store: dict, vec: np.ndarray, k: int = SIMILARITY_TOP_K):
    distances, ids = store["index"].search(vec.reshape(1, -1).astype("float32"), k)
    sf = []
    dists = []
    for raw_id, d in zip(ids[0], distances[0]):
        if raw_id < 0:
            continue
        doc_id = store["idx_to_id"][int(raw_id)]
        doc = store["docstore"]._dict[doc_id]
        md = doc.metadata or {}
        sf.append(md.get("source_file") or md.get("doc_name") or "?")
        dists.append(float(d))
    return sf, dists


def test_retrieval_snapshot_matches(saved_vectors, snapshot, vectorstores):
    vectors, labels = saved_vectors
    failures = []

    for vec, label in zip(vectors, labels):
        query = label["query"]
        domain = label["domain"]
        snap = snapshot.get(query)
        if snap is None:
            failures.append({
                "query": query, "domain": domain,
                "reason": f"missing snapshot entry; re-run save_test_vectors.py --update-snapshot",
            })
            continue

        actual_sf, actual_dists = _retrieve_topk(vectorstores[domain], vec,
                                                 k=SIMILARITY_TOP_K)

        snap_sf = list(snap["top_k_source_files"])
        snap_dists = list(snap["top_k_l2_distances"])

        actual_set, snap_set = set(actual_sf), set(snap_sf)
        added = sorted(actual_set - snap_set)
        removed = sorted(snap_set - actual_set)

        rank_drifts = []
        sorted_actual = sorted(actual_dists)
        sorted_snap = sorted(snap_dists)
        for rank, (a, s) in enumerate(zip(sorted_actual, sorted_snap), start=1):
            d = abs(a - s)
            if d >= L2_DRIFT_TOLERANCE:
                rank_drifts.append({"rank": rank, "actual": a, "snapshot": s, "abs_diff": d})

        if added or removed or rank_drifts:
            failures.append({
                "query": query, "domain": domain,
                "added_source_files": added,
                "removed_source_files": removed,
                "rank_drifts": rank_drifts,
                "actual_topk": list(zip(actual_sf, actual_dists)),
                "snapshot_topk": list(zip(snap_sf, snap_dists)),
            })

    if failures:
        lines = [f"\n{len(failures)} regression failure(s):"]
        for f in failures:
            lines.append(f"  query: {f['query']!r} (domain={f['domain']})")
            if f.get("added_source_files"):
                lines.append(f"    + added (live but not in snapshot): {f['added_source_files']}")
            if f.get("removed_source_files"):
                lines.append(f"    - removed (snapshot but not live):  {f['removed_source_files']}")
            for d in f.get("rank_drifts", []):
                lines.append(
                    f"    Δ rank={d['rank']}  actual={d['actual']:.4f}  "
                    f"snapshot={d['snapshot']:.4f}  |Δ|={d['abs_diff']:.4f}"
                )
            lines.append(f"    snapshot top-{SIMILARITY_TOP_K}: {f.get('snapshot_topk')}")
            lines.append(f"    live     top-{SIMILARITY_TOP_K}: {f.get('actual_topk')}")
        pytest.fail("\n".join(lines))


def test_every_query_returns_at_least_one_chunk(saved_vectors, vectorstores):
    MAX_L2 = 1.2
    vectors, labels = saved_vectors
    misses = []
    for vec, label in zip(vectors, labels):
        domain = label["domain"]
        query = label["query"]
        _sf, dists = _retrieve_topk(vectorstores[domain], vec, k=SIMILARITY_TOP_K)
        within = sum(1 for d in dists if d <= MAX_L2)
        if within == 0:
            misses.append((domain, query, min(dists)))
    assert not misses, (
        f"{len(misses)} regression query/queries returned 0 chunks within "
        f"MAX_L2_DISTANCE={MAX_L2}: {misses}"
    )
