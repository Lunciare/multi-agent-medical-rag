"""Wilson 95% CI utility shared across evaluation scripts.

Every Bernoulli proportion reported in `evaluate_routing.py`,
`evaluate_retrieval.py`, `evaluate_generation.py`, and
`evaluate_chunk_relevance.py` should be formatted via `fmt(k, n)` so the
terminal output and markdown summaries carry the same `X.X% [lo–hi]` shape
that the final report uses.

Confidence intervals are Wilson score intervals at α=0.05 (95% confidence),
computed by `statsmodels.stats.proportion.proportion_confint(method="wilson")`.
"""

from statsmodels.stats.proportion import proportion_confint


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Return (lo, hi) Wilson 95% CI bounds in [0, 1]. Returns (0, 0) when n=0."""
    if n == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def fmt(k: int, n: int) -> str:
    """Format a Bernoulli proportion as 'rate% [lo%–hi%]'.

    Uses U+2013 (en-dash) between the lo and hi bounds — matching the style
    already established in report_final.md §4.4, §4.7, and §4.8.
    """
    rate = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    return f"{rate:.1%} [{lo:.1%}–{hi:.1%}]"
