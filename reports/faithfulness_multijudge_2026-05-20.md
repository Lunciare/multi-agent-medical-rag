# Multi-Judge Faithfulness Evaluation (test split, n=70)

**Date:** 2026-05-20 11:06:59  
**Elapsed:** 373.9s (6.2 min)  
**Total judge calls:** 116  
**Raw per-case CSV:** [`faithfulness_multijudge_raw_2026-05-20.csv`](faithfulness_multijudge_raw_2026-05-20.csv)

## Configured Judges

| Role | Provider | Model URI | HTTP errors | Retries exhausted | Successful calls |
|---|---|---|---|---|---|
| yandex_primary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt/latest` | 0 | 0 | 58 |
| secondary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest` | 0 | 0 | 58 |

## Per-Judge Faithfulness (test split, Wilson 95% CI)

| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| yandex_primary | 58 | 58 | 0 | 100.0% | [93.8%–100.0%] |
| secondary | 58 | 58 | 0 | 100.0% | [93.8%–100.0%] |

## Minimum-Judge Faithfulness (all judges = FAITHFUL)

Cases where every available judge returned a non-None label: 58 / 58 (excluded 0 due to None from at least one judge).  
All judges agreed FAITHFUL on **58 / 58** cases = **100.0% [Wilson 95% CI 93.8%–100.0%]**.

## Pairwise Cohen's κ

| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |
|---|---|---|---|---|
| (yandex_primary, secondary) | 58 | 58 | nan | almost perfect |

## Disagreement Cases

0 case(s) where the configured judges did not all agree on a non-None label (excluding fallbacks).

| Case | Tier | Domain | yandex_primary | secondary |
|---|---|---|---|---|

## Tier 3 Fallback

12 / 15 Tier 3 cases returned the 'Insufficient evidence' fallback (excluded from judge totals).
