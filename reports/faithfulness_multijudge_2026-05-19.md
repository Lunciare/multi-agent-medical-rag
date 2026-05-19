# Multi-Judge Faithfulness Evaluation (test split, n=70)

**Date:** 2026-05-19 21:54:10  
**Elapsed:** 833.0s (13.9 min)  
**Total judge calls:** 140  
**Raw per-case CSV:** [`faithfulness_multijudge_raw_2026-05-19.csv`](faithfulness_multijudge_raw_2026-05-19.csv)

## Configured Judges

| Role | Provider | Model URI | HTTP errors | Retries exhausted | Successful calls |
|---|---|---|---|---|---|
| yandex_primary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt/latest` | 0 | 0 | 70 |
| secondary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest` | 0 | 0 | 70 |

## Per-Judge Faithfulness (test split, Wilson 95% CI)

| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| yandex_primary | 70 | 70 | 0 | 100.0% | [94.8%–100.0%] |
| secondary | 69 | 70 | 0 | 98.6% | [92.3%–99.7%] |

## Minimum-Judge Faithfulness (all judges = FAITHFUL)

Cases where every available judge returned a non-None label: 70 / 70 (excluded 0 due to None from at least one judge).  
All judges agreed FAITHFUL on **69 / 70** cases = **98.6% [Wilson 95% CI 92.3%–99.7%]**.

## Pairwise Cohen's κ

| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |
|---|---|---|---|---|
| (yandex_primary, secondary) | 70 | 69 | 0.000 | poor |

## Disagreement Cases

1 case(s) where the configured judges did not all agree on a non-None label (excluding fallbacks).

| Case | Tier | Domain | yandex_primary | secondary |
|---|---|---|---|---|
| cardio_40 | 2 | cardiologist | FAITHFUL | HALLUCINATION |

## Tier 3 Fallback

0 / 15 Tier 3 cases returned the 'Insufficient evidence' fallback (excluded from judge totals).
