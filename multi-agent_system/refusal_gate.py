"""Numeric out-of-scope refusal gate.

Replaces the prompt-only "Insufficient evidence" fallback (which `report_final.md`
§5.2 documented as a 0/16 failure on Tier 3) with a deterministic numeric check
in front of the LLM call.

Two candidate signals are implemented; the active signal is selected by the
`signal` argument and tuned offline against `golden_dev.json` by
`tests/tune_refusal_gate.py`:

  Signal A — minimum L2 distance:
      reject if min(distances over top-K retrieved chunks) > L2_REJECT_MIN.
      Simple, interpretable, single-parameter.

  Signal B — corpus-distance k-sigma rule:
      reject if min(distances) > μ_corpus − k · σ_corpus, where μ_corpus and
      σ_corpus are the mean and std of nearest-neighbour L2 distances over a
      random sample of 1000 in-corpus chunks (cached per specialty under
      data/processed/{specialty}/corpus_dist_stats.json).
      Intuition: in-distribution queries should fall *inside* the typical
      near-neighbour distance scale of the corpus.

If `corpus_dist_stats` is omitted Signal B falls back to never rejecting (so
callers can pass only `l2_reject_min` and still get a usable gate, which the
smoke test depends on).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    """Empirical L2 distribution of all-pairs chunk-to-chunk distances over a
    random sample of `sample_size` in-corpus chunks. Returns mean and std.

    Why all-pairs and not nearest-neighbour: in this 256-d, pre-normalised
    Yandex embedding space, *nearest-neighbour* chunk distances are very small
    (μ ≈ 0.37) because chunks from the same document cluster tightly. Query→doc
    distances live in a different (larger) regime (typically 0.85–1.30), so a
    nearest-neighbour μ would give an unreachably strict Signal B threshold.
    The all-pairs distance distribution captures the *typical* inter-chunk L2
    scale (μ ≈ 1.4 for randomly oriented 256-d unit vectors), which gives a
    useful k-sigma threshold for query-time use. Documented in §4.5.
    """
    rng = random.Random(random_seed)
    ntotal = vectorstore.index.ntotal
    if ntotal == 0:
        raise ValueError("empty vectorstore — cannot compute corpus distance stats")
    sample_n = min(sample_size, ntotal)
    indices = rng.sample(range(ntotal), sample_n)

    vecs = np.stack([vectorstore.index.reconstruct(int(i)) for i in indices]).astype("float32")
    # All-pairs squared L2 via expanded dot products: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b.
    # For our pre-normalised vectors ||a||^2 ≈ 1, so ||a-b||^2 ≈ 2 - 2 cos(a,b).
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
    """Cache-aware accessor for chunk-to-chunk distance stats.

    Reads cache_path if present; otherwise computes once, writes to disk, and
    returns. Used by RefusalGate.from_vectorstore so the gate construction is
    O(1) on warm starts.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                stats = CorpusDistStats.from_dict(json.load(f))
            if stats.n >= sample_size // 2:
                return stats
        except (json.JSONDecodeError, KeyError):
            pass

    stats = compute_corpus_dist_stats(vectorstore, sample_size=sample_size,
                                      random_seed=random_seed)
    stats.specialty = specialty
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)
    return stats


class RefusalGate:
    """Numeric out-of-scope gate. Returns True from `refuse(query)` when the
    query should be rejected before reaching the LLM.

    Construction:
      gate = RefusalGate(
          vectorstore,
          l2_reject_min=1.30,             # Signal A threshold (None disables it)
          corpus_dist_stats={"mu": 1.2, "sigma": 0.15},   # Signal B μ, σ
          corpus_dist_k=2.0,              # Signal B k (None disables it)
          signal="A",                     # "A" or "B" — selects which test fires
          top_k=5,
      )

    `refuse()` retrieves the top-K chunks for the query and applies the
    selected signal's threshold to the minimum L2 distance.
    """

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

    # ----- query-time API -----

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
        """Return True if the query should be refused (out-of-scope)."""
        if not query or not query.strip():
            return True
        min_dist = self._min_dist(query)
        if min_dist is None:
            return True  # nothing retrieved — refuse
        if self.signal == SIGNAL_A_MIN_L2:
            return self.signal_a_rejects(min_dist)
        return self.signal_b_rejects(min_dist)

    # ----- diagnostics -----

    def explain(self, query: str) -> dict:
        """Return a dict with both signals' verdicts and the min distance.
        Used by the tuner to grid-search thresholds without redoing retrieval.
        """
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
        """Construct a gate with the corpus distance cache populated lazily."""
        cache_path = os.path.join(processed_dir, "corpus_dist_stats.json")
        stats = load_or_compute_corpus_dist_stats(
            vectorstore, cache_path=cache_path, specialty=specialty,
            sample_size=sample_size,
        )
        return cls(vectorstore=vectorstore, l2_reject_min=l2_reject_min,
                   corpus_dist_stats=stats.to_dict(),
                   corpus_dist_k=corpus_dist_k, signal=signal, top_k=top_k)
