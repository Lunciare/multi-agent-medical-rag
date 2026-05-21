"""Offline retrieval regression test.

For each of the 10 canonical regression queries (5 cardiology + 5 endocrinology),
this test:

  1. Loads the pre-saved query embedding from `test_vectors.npy` (so the test
     never calls the live Yandex embedding API).
  2. Runs that vector through the canonical FAISS index for the correct domain.
  3. Compares the resulting top-K=5 (source_files, L2 distances) against the
     snapshot stored in `test_retrieval_snapshot.json`.

Two assertions per query:

  - **Set equality on source files.** The set of retrieved `source_file`
    values must equal the snapshot's set. Drift in *which documents* surface
    in the top-5 (chunking change, re-embedding, score-tie reordering across
    chunks of the same doc, …) fails the test.
  - **Per-rank L2 distance drift ≤ 0.1.** The retrieved distances and the
    snapshot distances are matched rank-by-rank (both sorted ascending) and
    each pair must differ by less than 0.1 absolute. This catches embedding
    drift that doesn't swap any document but does shift the scoring.

If either assertion fails, the test prints the diff (added / removed source
files and the per-rank distance differences) and the offending query id so the
operator can decide whether the change is intentional (and re-run
`save_test_vectors.py --update-snapshot`) or a regression to investigate.
"""

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
    """Load each domain's FAISS index using raw `faiss.read_index` plus the
    pickled `(InMemoryDocstore, idx_to_id)` sidecar.

    The sibling test files in this directory (`test_safety.py`,
    `test_integration.py`, `test_error_handling.py`) rely on `conftest.py`
    installing `MagicMock` over `langchain_community.vectorstores` so they
    can patch FAISS without dragging in heavy dependencies. That mocking would
    break a `FAISS.load_local(...)` call here, so the regression test bypasses
    LangChain entirely: `faiss.read_index` gives us the raw vectors, the pkl
    sidecar gives us the docstore + index→docstore-id map, and we resolve
    each retrieved numeric id to a `Document` ourselves. Result: the metadata
    chain (`source_file`, `doc_name`) is identical to what `Agent.answer()`
    would see at runtime, without needing LangChain unmocked.
    """
    import pickle, sys as _sys

    # `conftest.py` installs `MagicMock` over the langchain modules so the
    # sibling tests can patch FAISS without the real lib. Unpickling the
    # docstore requires the real `langchain_community.docstore.in_memory`
    # module — pop the mocks so the real lib gets imported on next access.
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
    """Top-K retrieval via raw faiss; returns parallel lists (source_files,
    l2_distances). FAISS's IndexFlatL2 returns *squared* L2 distances, and
    LangChain's `similarity_search_with_score_by_vector` (used by
    save_test_vectors.py to write the snapshot) passes those through
    verbatim — so we report the same raw squared-L2 here for an apples-to-
    apples comparison. The production code's `MAX_L2_DISTANCE = 1.2` is the
    same squared-L2 quantity (the codebase labels it "L2" loosely)."""
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
    """For every canonical query, the live top-K set must match the snapshot
    set, and per-rank L2 distance drift must be < 0.1."""
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
    """Sanity: every canonical query must retrieve at least one chunk within
    `MAX_L2_DISTANCE=1.2`. Catches a different failure mode than the snapshot
    test (e.g. a corrupted index that returns garbage distances all > 1.2).
    Distances are squared L2 — matching the production codebase's convention."""
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
