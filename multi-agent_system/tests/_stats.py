
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def fmt(k: int, n: int) -> str:
    rate = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    return f"{rate:.1%} [{lo:.1%}–{hi:.1%}]"
