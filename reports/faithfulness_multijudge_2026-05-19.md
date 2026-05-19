# Multi-Judge Faithfulness Evaluation (test split, n=70)

**Date:** 2026-05-19 23:04:28  
**Elapsed:** 354.5s (5.9 min)  
**Total judge calls:** 116  
**Raw per-case CSV:** [`faithfulness_multijudge_raw_2026-05-19.csv`](faithfulness_multijudge_raw_2026-05-19.csv)

## Configured Judges

| Role | Provider | Model URI | HTTP errors | Retries exhausted | Successful calls |
|---|---|---|---|---|---|
| yandex_primary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt/latest` | 0 | 0 | 58 |
| secondary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest` | 0 | 0 | 58 |

## Per-Judge Faithfulness (test split, Wilson 95% CI)

| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| yandex_primary | 58 | 58 | 0 | 100.0% | [93.8%–100.0%] |
| secondary | 57 | 58 | 0 | 98.3% | [90.9%–99.7%] |

## Minimum-Judge Faithfulness (all judges = FAITHFUL)

Cases where every available judge returned a non-None label: 58 / 58 (excluded 0 due to None from at least one judge).  
All judges agreed FAITHFUL on **57 / 58** cases = **98.3% [Wilson 95% CI 90.9%–99.7%]**.

## Pairwise Cohen's κ

| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |
|---|---|---|---|---|
| (yandex_primary, secondary) | 58 | 57 | 0.000 | poor |

## Disagreement Cases

1 case(s) where the configured judges did not all agree on a non-None label (excluding fallbacks).

| Case | Tier | Domain | yandex_primary | secondary |
|---|---|---|---|---|
| cardio_40 | 2 | cardiologist | FAITHFUL | HALLUCINATION |

## Tier 3 Fallback

12 / 15 Tier 3 cases returned the 'Insufficient evidence' fallback (excluded from judge totals).
