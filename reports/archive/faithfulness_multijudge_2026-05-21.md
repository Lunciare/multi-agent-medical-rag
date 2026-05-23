# Multi-Judge Faithfulness Evaluation (test split, n=140)

**Date:** 2026-05-21 21:19:24  
**Elapsed:** 911.2s (15.2 min)  
**Total judge calls:** 266  
**Raw per-case CSV:** [`faithfulness_multijudge_raw_2026-05-21.csv`](faithfulness_multijudge_raw_2026-05-21.csv)

## Configured Judges

| Role | Provider | Model URI | Auth | Rate-limit | Conn | Timeout | Other | Successes |
|---|---|---|---|---|---|---|---|---|
| yandex_primary | yandex | `gpt://b1gmsujt9mgtp0qubblu/yandexgpt/latest` | 0 | 0 | 0 | 0 | 0 | 132 |
| secondary | yandex | `gpt://b1gmsujt9mgtp0qubblu/yandexgpt-lite/latest` | 0 | 0 | 0 | 0 | 0 | 132 |

## Per-Judge Faithfulness (test split, Wilson 95% CI)

| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| yandex_primary | 132 | 132 | 1 | 100.0% | [97.2%–100.0%] |
| secondary | 131 | 132 | 1 | 99.2% | [95.8%–99.9%] |

## Minimum-Judge Faithfulness (all judges = FAITHFUL)

Cases where every available judge returned a non-None label: 132 / 133 (excluded 1 due to None from at least one judge).  
All judges agreed FAITHFUL on **131 / 132** cases = **99.2% [Wilson 95% CI 95.8%–99.9%]**.

## Pairwise Cohen's κ

| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |
|---|---|---|---|---|
| (yandex_primary, secondary) | 132 | 131 | 0.000 | degenerate (one marginal = 0; observed agreement = 131/132) |

## Disagreement Cases

1 case(s) where the configured judges did not all agree on a non-None label (excluding fallbacks).

| Case | Tier | Domain | yandex_primary | secondary |
|---|---|---|---|---|
| cardio_40 | 2 | cardiologist | FAITHFUL | HALLUCINATION |

## Tier 3 Fallback

7 / 29 Tier 3 cases returned the 'Insufficient evidence' fallback (excluded from judge totals).
