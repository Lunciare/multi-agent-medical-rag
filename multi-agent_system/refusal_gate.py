
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SIGNAL_A_MIN_L2 = "A"
SIGNAL_B_CORPUS_K = "B"

DEFAULT_TOP_K = 5
DEFAULT_CORPUS_SAMPLE = 1000


@dataclass
class CorpusDistStats:
    mu: float
    sigma: float
    n: int = 0
    specialty: Optional[str] = None

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n,
                "specialty": self.specialty}

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusDistStats":
        return cls(mu=float(d["mu"]), sigma=float(d["sigma"]),
                   n=int(d.get("n", 0)), specialty=d.get("specialty"))


def compute_corpus_dist_stats(vectorstore, *, sample_size: int = DEFAULT_CORPUS_SAMPLE,
                              random_seed: int = 42) -> CorpusDistStats:
    rng = random.Random(random_seed)
    ntotal = vectorstore.index.ntotal
    if ntotal == 0:
        raise ValueError("empty vectorstore — cannot compute corpus distance stats")
    sample_n = min(sample_size, ntotal)
    indices = rng.sample(range(ntotal), sample_n)

    vecs = np.stack([vectorstore.index.reconstruct(int(i)) for i in indices]).astype("float32")
    sq_norms = np.sum(vecs * vecs, axis=1)
    G = vecs @ vecs.T
    pair_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G
    iu = np.triu_indices(sample_n, k=1)
    pair_l2 = np.sqrt(np.maximum(pair_sq[iu], 0.0))
    return CorpusDistStats(mu=float(pair_l2.mean()),
                           sigma=float(pair_l2.std()),
                           n=int(sample_n))


def load_or_compute_corpus_dist_stats(vectorstore, *, cache_path: str,
                                      specialty: Optional[str] = None,
                                      sample_size: int = DEFAULT_CORPUS_SAMPLE,
                                      random_seed: int = 42) -> CorpusDistStats:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                stats = CorpusDistStats.from_dict(json.load(f))
            if stats.n >= sample_size // 2:
                return stats
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not read corpus-dist cache at %s (%s: %s); "
                           "recomputing.", cache_path, type(exc).__name__, exc)

    stats = compute_corpus_dist_stats(vectorstore, sample_size=sample_size,
                                      random_seed=random_seed)
    stats.specialty = specialty

    cache_dir = os.path.dirname(cache_path) or "."
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2)
        os.replace(tmp_path, cache_path)
    except OSError as exc:
        logger.warning("Could not persist corpus-dist cache at %s (%s: %s); "
                       "returning in-memory stats only.", cache_path,
                       type(exc).__name__, exc)
        try:
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    return stats


class RefusalGate:

    def __init__(self,
                 vectorstore,
                 *,
                 l2_reject_min: Optional[float] = None,
                 corpus_dist_stats: Optional[dict] = None,
                 corpus_dist_k: float = 1.0,
                 signal: str = SIGNAL_A_MIN_L2,
                 top_k: int = DEFAULT_TOP_K):
        if signal not in (SIGNAL_A_MIN_L2, SIGNAL_B_CORPUS_K):
            raise ValueError(f"signal must be {SIGNAL_A_MIN_L2!r} or "
                             f"{SIGNAL_B_CORPUS_K!r}, got {signal!r}")
        self.vectorstore = vectorstore
        self.l2_reject_min = l2_reject_min
        self.corpus_dist_stats = corpus_dist_stats or {}
        self.corpus_dist_k = corpus_dist_k
        self.signal = signal
        self.top_k = top_k


    def _min_dist(self, query: str) -> Optional[float]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        if not docs_and_scores:
            return None
        return min(score for _doc, score in docs_and_scores)

    def signal_a_rejects(self, min_dist: float) -> bool:
        if self.l2_reject_min is None:
            return False
        return min_dist > self.l2_reject_min

    def signal_b_rejects(self, min_dist: float) -> bool:
        if not self.corpus_dist_stats:
            return False
        mu = float(self.corpus_dist_stats.get("mu", 0.0))
        sigma = float(self.corpus_dist_stats.get("sigma", 0.0))
        threshold = mu - self.corpus_dist_k * sigma
        return min_dist > threshold

    def refuse(self, query: str) -> bool:
        if not query or not query.strip():
            return True
        min_dist = self._min_dist(query)
        if min_dist is None:
            return True
        if self.signal == SIGNAL_A_MIN_L2:
            return self.signal_a_rejects(min_dist)
        return self.signal_b_rejects(min_dist)


    def explain(self, query: str) -> dict:
        min_dist = self._min_dist(query)
        return {
            "min_dist": min_dist,
            "signal_a_rejects": self.signal_a_rejects(min_dist) if min_dist is not None else True,
            "signal_b_rejects": self.signal_b_rejects(min_dist) if min_dist is not None else True,
            "active_signal": self.signal,
            "l2_reject_min": self.l2_reject_min,
            "corpus_dist_stats": self.corpus_dist_stats,
            "corpus_dist_k": self.corpus_dist_k,
        }

    @classmethod
    def from_vectorstore(cls,
                         vectorstore,
                         *,
                         specialty: str,
                         processed_dir: str,
                         l2_reject_min: Optional[float] = None,
                         corpus_dist_k: float = 1.0,
                         signal: str = SIGNAL_A_MIN_L2,
                         top_k: int = DEFAULT_TOP_K,
                         sample_size: int = DEFAULT_CORPUS_SAMPLE) -> "RefusalGate":
        cache_path = os.path.join(processed_dir, "corpus_dist_stats.json")
        stats = load_or_compute_corpus_dist_stats(
            vectorstore, cache_path=cache_path, specialty=specialty,
            sample_size=sample_size,
        )
        return cls(vectorstore=vectorstore, l2_reject_min=l2_reject_min,
                   corpus_dist_stats=stats.to_dict(),
                   corpus_dist_k=corpus_dist_k, signal=signal, top_k=top_k)
