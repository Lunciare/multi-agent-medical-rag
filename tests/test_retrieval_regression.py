import json
from pathlib import Path

import faiss
import numpy as np
import pytest

SIMILARITY_TOP_K = 5
MAX_L2_DISTANCE = 1.2

REPO_ROOT = Path(__file__).resolve().parent.parent
VECTORS_PATH = REPO_ROOT / "multi-agent_system" / "tests" / "data" / "test_vectors.npy"
LABELS_PATH = REPO_ROOT / "multi-agent_system" / "tests" / "data" / "test_vector_labels.json"

FAISS_PATHS = {
    "cardiology": REPO_ROOT / "data" / "processed" / "cardiology" / "faiss_index" / "index.faiss",
    "endocrinology": REPO_ROOT / "data" / "processed" / "endocrinology" / "faiss_index" / "index.faiss",
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
def faiss_indices():
    indices = {}
    for domain, path in FAISS_PATHS.items():
        _require(path, f"{domain} FAISS index")
        indices[domain] = faiss.read_index(str(path))
    return indices


def test_every_query_returns_at_least_one_chunk(saved_vectors, faiss_indices):
    vectors, labels = saved_vectors

    regressions = []
    for vec, label in zip(vectors, labels):
        domain = label["domain"]
        query = label["query"]
        index = faiss_indices[domain]

        distances, _ids = index.search(vec.reshape(1, -1), SIMILARITY_TOP_K)
        min_distance = float(distances[0].min())
        hits = int((distances[0] <= MAX_L2_DISTANCE).sum())

        if hits == 0:
            print(f"REGRESSION: {query} returned 0 chunks. Check MAX_L2_DISTANCE.")
            regressions.append((domain, query, min_distance))

    assert not regressions, (
        f"{len(regressions)} query/queries retrieved 0 chunks within "
        f"MAX_L2_DISTANCE={MAX_L2_DISTANCE}: {regressions}"
    )
