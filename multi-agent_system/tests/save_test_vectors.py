"""One-time script: regenerates tests/data/test_vectors.npy + test_vector_labels.json
from the live Yandex embedding API.

Requires YANDEX_API_KEY / YANDEX_PROJECT_ID (loaded from multi-agent_system/.env).
Run this ONCE; the produced .npy and .json files are committed to the repo so
that tests/test_retrieval_regression.py runs offline thereafter.

Usage:
    cd multi-agent_system
    python tests/save_test_vectors.py
"""
import json
import sys
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


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = YandexNativeEmbeddings()
    pairs = (
        [(q, "cardiology") for q in CARDIOLOGY_QUERIES]
        + [(q, "endocrinology") for q in ENDOCRINOLOGY_QUERIES]
    )

    vectors = []
    labels = []
    for query, domain in pairs:
        print(f"Embedding [{domain}] '{query}'...")
        vectors.append(embedder.embed_query(query))
        labels.append({"query": query, "domain": domain})

    arr = np.asarray(vectors, dtype=np.float32)

    vectors_path = out_dir / "test_vectors.npy"
    labels_path = out_dir / "test_vector_labels.json"

    np.save(vectors_path, arr)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"\nSaved {arr.shape} vectors → {vectors_path}")
    print(f"Saved {len(labels)} labels  → {labels_path}")


if __name__ == "__main__":
    main()
