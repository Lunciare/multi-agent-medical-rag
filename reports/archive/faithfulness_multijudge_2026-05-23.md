# Multi-Judge Faithfulness Evaluation (test split, n=140) — 3-judge edition

**Date:** 2026-05-23
**Elapsed:** 1766.9s (29.4 min) on the tertiary-only re-run
**Total judge calls:** 280 Yandex (primary + secondary, reused from the
2026-05-21 run) + 133 OpenRouter tertiary = **413**
**Raw per-case CSV:** [`faithfulness_multijudge_raw_2026-05-23.csv`](faithfulness_multijudge_raw_2026-05-23.csv)
**Tertiary-judge driver:** [`multi-agent_system/tests/run_tertiary_judge.py`](../multi-agent_system/tests/run_tertiary_judge.py)

This run extends the Stage-39 2-judge multi-judge evaluation (`faithfulness_multijudge_raw_2026-05-21.csv`) with a third, cross-vendor judge: OpenAI's open-weights GPT-OSS-120B served via OpenRouter's free tier. The Yandex primary + secondary verdicts are reused verbatim from the 2026-05-21 CSV (no re-pay for those 280 calls); the new column is the tertiary verdict alone, on the same 133 non-fallback cases. This closes the residual same-vendor circularity flagged in `report_final.md` §5.3 and §6 L6.

## Configured Judges

| Role | Provider | Model URI |
|---|---|---|
| Primary | yandex | `gpt://{folder}/yandexgpt/latest` |
| Secondary | yandex | `gpt://{folder}/yandexgpt-lite/latest` |
| Tertiary (cross-vendor) | openrouter | `openai/gpt-oss-120b:free` |

> **Model substitution note.** The independent audit (Part E §G.2) proposed `meta-llama/llama-3.1-8b-instruct:free` as the tertiary judge. OpenRouter retired that slug between the audit being written (2026-05-22) and this run (2026-05-23); the working free-tier replacement with available capacity is OpenAI's open-weights GPT-OSS-120B, which is a larger, stronger judge in the same cross-vendor-vs-Yandex slot. Either model fulfils the audit's stated goal of breaking the same-family circularity bound.

## Per-Judge Faithfulness (3-judge intersection, n=132)

| Judge | Faithful | Total Judged | Rate | Wilson 95% CI |
|---|---|---|---|---|
| yandex_primary | 132 | 132 | 100.0% | [97.2%–100.0%] |
| secondary | 131 | 132 | 99.2% | [95.8%–99.9%] |
| tertiary (cross-vendor) | 128 | 132 | 97.0% | [92.5%–98.8%] |
| **Minimum (all 3 agree FAITHFUL)** | **127** | **132** | **96.2%** | **[91.4%–98.4%]** |

The n=132 denominator is the case intersection where all three judges returned a non-fallback, non-None verdict. Of the original 140 test-split cases, 7 triggered the Stage-7 refusal-gate fallback (3 gastro T3 + 4 infect T3) and 1 (`infect_17`) carried a None primary verdict from the 2026-05-21 Yandex run; both groups are excluded.

## Pairwise Agreement (Cohen's κ and Gwet's AC1)

Cohen's κ is degenerate when one rater's marginal class probability is 0, since chance agreement equals observed agreement and the denominator vanishes. Gwet's AC1 stays well-defined: it uses the empirical class prior averaged across raters, which only collapses when both raters tie on the same extreme. Report both; AC1 is the better-behaved chance-corrected statistic when one marginal collapses. The Landis & Koch band is computed from AC1 in the degenerate-κ rows.

| Pair | n | Agreements | Cohen's κ | Gwet's AC1 | Landis & Koch |
|---|---:|---:|---:|---:|---|
| (yandex_primary, secondary) | 132 | 131 | 0.000 | 0.992 | degenerate κ (primary marginal = 0); via AC1: almost perfect |
| (yandex_primary, tertiary) | 132 | 128 | 0.000 | 0.969 | degenerate κ (primary marginal = 0); via AC1: almost perfect |
| (secondary, tertiary) | 132 | 127 | −0.012 | 0.961 | less than chance (κ); almost perfect (AC1) |

The (secondary, tertiary) κ = −0.012 is technically negative — but the disagreement structure is the diagnostic insight, not a model bug. Secondary's only HALLUCINATION verdict is `cardio_40`; tertiary's HALLUCINATION verdicts are `{cardio_17, endo_21, endo_38, infect_34}`. The two HALLUCINATION sets are **disjoint** — they flag different cases — so Cohen's κ correctly reports "no agreement structure beyond chance about which cases are hallucinations". AC1 = 0.961 simultaneously reports "but they still agree 96% of the time overall", which is what you'd predict from each judge independently flagging ~1–4% of cases as HALL. Both statistics are mathematically correct and tell complementary stories.

## Disagreement cases (5 of 132)

| Case ID | Tier | Domain | Primary | Secondary | Tertiary | Pattern |
|---|---|---|---|---|---|---|
| `cardio_17` | 1 | cardiology | FAITHFUL | FAITHFUL | HALLUCINATION | tertiary-only (new in this run) |
| `cardio_40` | 2 | cardiology | FAITHFUL | HALLUCINATION | FAITHFUL | secondary-only (Stage-31 historical case; see §5.3) |
| `endo_21` | 1 | endocrinology | FAITHFUL | FAITHFUL | HALLUCINATION | tertiary-only (new in this run) |
| `endo_38` | 2 | endocrinology | FAITHFUL | FAITHFUL | HALLUCINATION | tertiary-only (new in this run) |
| `infect_34` | 1 | infectiology | FAITHFUL | FAITHFUL | HALLUCINATION | tertiary-only (new in this run) |

## Interpretation

**Drop from the 2-judge headline: 99.2% (131/132) → 96.2% (127/132) = −3.0 pp.** This sits squarely inside the audit's predicted 3–8 pp range, derived from Zheng et al.'s \cite{zheng2023mtbench} measurement of 5–25 pp same-family judge self-preference inflation, applied to a 99.2% ceiling. The cross-vendor judge surfaced 4 cases (`cardio_17`, `endo_21`, `endo_38`, `infect_34`) where the Yandex pair both said FAITHFUL but a model from a different vendor family said HALLUCINATION — the empirical evidence of same-vendor blind-spot.

**The faithfulness claim now has both bounds.** The 2-judge min-judge rate of 99.2% remains valid as the *upper bound* (under same-family bias). The 3-judge min-judge rate of 96.2% [91.4%–98.4%] is now the production-quality *lower bound*: a case is only counted FAITHFUL if a model from a different vendor agrees with both Yandex judges. The Wilson lower bound on the 3-judge rate is 91.4% — still very high by RAG-faithfulness standards, but with a credibility floor that the 2-judge headline did not have.

## Notes on the run

- `MAX_RETRIES` was temporarily reduced from 5 → 1 inside the tertiary-only driver so OpenRouter throttling would fail fast (write `NONE` and proceed) rather than block the per-case loop. In the actual run no tertiary call returned `NONE` and no retries were exhausted; GPT-OSS-120B was unsaturated at the time of the run.
- The driver writes the output CSV per case with `fsync` after each row, so the run is resumable: re-launching `run_tertiary_judge.py` skips cases whose `id` is already in the output CSV. An earlier attempt at the full 3-judge `evaluate_generation.py --mode multi_judge` was stalled by OpenRouter rate-limiting on `meta-llama/llama-3.3-70b-instruct:free`; the dedicated tertiary-only driver was written precisely to bypass that failure mode.
- The earlier 2-judge markdown summary (`reports/archive/faithfulness_multijudge_2026-05-21.md`) is retained for traceability but is superseded by this file.
