# Stage 5 Report: Multi-Judge Faithfulness Evaluation

**Date:** 2026-05-19

## 1. What Was Changed
- `multi-agent_system/settings.py`: added `PRIMARY_JUDGE_PROVIDER` constant plus env-driven `SECONDARY_JUDGE_PROVIDER` and `TERTIARY_JUDGE_PROVIDER` (both default `None`). Added `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `SECONDARY_JUDGE_API_KEY` for the optional non-Yandex paths.
- `multi-agent_system/judges.py` (new): generic `judge_faithfulness(query, context, generated_answer, judge_config) -> Optional[bool]`. Parses `yandex:...`, `openrouter:...`, and `http:...` URI prefixes; reuses the existing Yandex `OpenAI` client for Yandex and `requests` for the OpenAI-compatible HTTP paths. Identical `JUDGE_SYSTEM_PROMPT` is exported and used by every judge so any rate difference is attributable to the model, not the prompt. Retries on 429/5xx with exponential backoff up to 5 attempts; throttles OpenRouter / generic-http judges to 2s between requests; returns `None` on persistent failure (the caller logs the failure with the case id and never silently substitutes another judge).
- `multi-agent_system/tests/evaluate_generation.py`: refactored. New `--mode {multi_judge,yandex_only}` argument (default `multi_judge`). Multi-judge mode: invokes every configured judge per case, writes a raw per-case CSV (`reports/faithfulness_multijudge_raw_<date>.csv`), and a markdown summary (`reports/faithfulness_multijudge_<date>.md`) containing per-judge Wilson 95% CI, pairwise Cohen's κ, the minimum-judge rate, per-judge HTTP-error counts, and the disagreement list. Fails clearly with `exit 2` if `SECONDARY_JUDGE_PROVIDER` is unset. `yandex_only` mode preserves the original single-judge implementation for backward compatibility during development.
- `multi-agent_system/tests/inspect_judge_disagreements.py` (new): read-only diagnostic helper that, given the raw CSV, replays retrieval + generation for each disagreement case and writes a markdown report with query, first 500 chars of context, generated answer, and each judge's verdict (`reports/judge_disagreement_inspection_<date>.md`).
- `README.md`: updated the Running Evaluations section to mark `multi_judge` as the default mode, document the new env-var configuration (Yandex same-vendor different-family and the optional cross-vendor slots), and add the disagreement-inspector command.
- `reports/report_final.md`: rewrote §4.4, §5.3, §6 Limitation 6, §7 (verbatim text reproduced in §6 below).
- `reports/faithfulness_multijudge_2026-05-19.md`, `reports/faithfulness_multijudge_raw_2026-05-19.csv`, `reports/judge_disagreement_inspection_2026-05-19.md`, `reports/multijudge_run_2026-05-19.log` (new artefacts from the run described in §4 below).

## 2. Judge Identifiers Used

| Role | Vendor | Model URI |
|---|---|---|
| Primary (`yandex_primary`) | yandex | `gpt://${YANDEX_PROJECT_ID}/yandexgpt/latest` |
| Secondary | yandex | `gpt://${YANDEX_PROJECT_ID}/yandexgpt-lite/latest` |
| Tertiary | — | **not configured for this run** |

Path A (free-tier cross-vendor judge — OpenRouter or any non-Yandex provider) was *not accessible* from the project's current API credentials; the supporting code is in `multi-agent_system/judges.py` (`openrouter:...` and `http:...` URI schemes) and `multi-agent_system/settings.py` (`OPENROUTER_API_KEY`, `SECONDARY_JUDGE_API_KEY`). Path B was used: a same-vendor different-family judge via the existing Yandex API key. Yandex's `llama/*` models were probed (`gpt://{folder}/llama/latest`, `gpt://{folder}/llama-lite/latest`, `gpt://{folder}/llama-70b/latest`, `gpt://{folder}/llama-8b/latest`, `gpt://{folder}/llama/rc`) — every llama variant returned `HTTP 400 — Failed to get model` on this folder, so the tertiary slot was left empty. The script's optional-tertiary contract handles this cleanly: zero tertiary rows, no fabricated values.

## 3. Per-Judge Faithfulness on the Test Split (Wilson 95% CI)

| Judge | Faithful | Total Judged | None | Rate | Wilson 95% CI |
|---|---|---|---|---|---|
| `yandex_primary` (`yandexgpt/latest`) | 70 | 70 | 0 | **100.0%** | **[94.8% – 100.0%]** |
| `secondary` (`yandexgpt-lite/latest`) | 69 | 70 | 0 | **98.6%** | **[92.3% – 99.7%]** |
| Tertiary | — | — | — | — | — |

Confidence intervals computed via `statsmodels.stats.proportion.proportion_confint(..., method="wilson")` at α=0.05.

## 4. Pairwise Cohen's κ and Landis & Koch Interpretation

| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |
|---|---|---|---|---|
| (`yandex_primary`, `secondary`) | 70 | 69 | **0.000** | **poor** (<0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 almost perfect) |
| (`yandex_primary`, `tertiary`) | — | — | n/a | tertiary not configured |
| (`secondary`, `tertiary`) | — | — | n/a | tertiary not configured |

**Caveat on the κ=0.000 value:** this is mathematically degenerate, not an evidence-light agreement. The primary judge marks every test case FAITHFUL, so its row marginal `P(HALLUCINATION) = 0`. With one rater's marginal at zero, expected agreement under marginal independence equals the observed agreement (both 69/70), forcing κ to exactly 0 regardless of how the disagreements distribute. A more informative κ would require either a judge that occasionally rejects what the primary accepts (the secondary did, but only once — pulling its `P(HALLUCINATION)` to 1/70 doesn't move the joint marginal enough) or a much larger evaluation set. We report 0.000 verbatim and flag this explicitly, rather than substituting a more flattering statistic.

## 5. Minimum-Judge Faithfulness (the new §7 headline)

A case counts as FAITHFUL only when every available judge returns `FAITHFUL`. Cases where any judge returned `None` are excluded; the exclusion count is reported below.

| Field | Value |
|---|---|
| Available judges agreeing FAITHFUL | **69 / 70** |
| Minimum-judge faithfulness rate | **98.6%** |
| Wilson 95% CI | **[92.3% – 99.7%]** |
| **Conservative lower bound quoted in §7** | **92.3%** |
| Cases excluded due to None from any judge | **0** |
| Tier 3 cases triggering "Insufficient evidence" fallback (excluded from judge totals) | 0 / 15 |

## 6. Per-Judge HTTP-Error and Exhausted-Retry Counts

| Judge | HTTP errors | Retries exhausted | Successful calls | None returns reaching the aggregator |
|---|---|---|---|---|
| `yandex_primary` | 0 | 0 | 70 | 0 |
| `secondary` | 0 | 0 | 70 | 0 |

Total judge calls across the run: 140. Total wall-clock: 833.0 s (13.9 min). No retries were exhausted, so no case was excluded from the minimum-judge rate computation.

## 7. Judge-Disagreement Cases

| Case | Tier | Domain | `yandex_primary` verdict | `secondary` verdict |
|---|---|---|---|---|
| `cardio_40` | 2 | cardiologist | **FAITHFUL** | **HALLUCINATION** |

Full diagnostic dump (query, first 500 chars of retrieved context, generated answer, both verdicts): [`reports/judge_disagreement_inspection_2026-05-19.md`](judge_disagreement_inspection_2026-05-19.md).

### Qualitative disagreement pattern

The single disagreement — `cardio_40` — exhibits the exact disagreement axis the multi-judge design was built to surface. The query asks for the likely diagnosis of a 30-year-old male with resuscitated out-of-hospital cardiac arrest, prolonged QTc 510 ms, and a sister with a similar event at 25, a presentation that strongly suggests *congenital* long QT syndrome. The retrieved context centres on a tangentially related case (a 30-something woman with seizure activity and a prolonged QTc of 500–530 ms leading to Torsades de Pointes) whose prolongation is explicitly attributed to herbal-remedy-induced *acquired* LQTS, with a passing remark that "normal QTc does not exclude congenital LQTS." The generated answer paraphrases the related case, then infers congenital LQTS for the new patient citing the family history. The flagship `yandex_primary` (yandexgpt/latest) judge accepts this inference as a faithful paraphrase plus a logical-inference move permitted by the judge prompt. The smaller `secondary` (yandexgpt-lite/latest) judge — given the same prompt — applies a stricter near-token-grounding standard and rejects the inference because the specific diagnosis "congenital long QT syndrome" is not directly stated in the retrieved context. The pattern is therefore: **the flagship judge treats reasonable clinical inference from related-but-distinct context as faithful; the lite judge requires the specific diagnosis label to appear in the retrieved tokens before allowing a FAITHFUL verdict**. Both verdicts are defensible — neither is unambiguously wrong — but the existence of the disagreement places an honest empirical lower bound on faithfulness at 98.6% / 92.3% (Wilson lower) instead of the primary judge's apparent 100% ceiling.

## 8. Verbatim Text of report_final.md §4.4 and §7

The following sections were inserted into `report_final.md` and are reproduced here verbatim.

### 8.1 §4.4 Faithfulness (Generation Quality)

> The full RAG pipeline (retrieval → LLM generation) is now evaluated by two independent LLM-as-a-judge models, each given the identical strict faithfulness prompt to keep the comparison clean. The primary judge is YandexGPT — the same model family used for generation. The secondary judge is YandexGPT-Lite, a distinct Yandex model not used anywhere in the generation pipeline; it breaks the strictest same-model circularity and lets us measure whether the smaller, faster Yandex judge applies a different token-grounding standard than the flagship. A third cross-vendor judge slot (e.g. an OpenRouter free-tier model) is supported by `evaluate_generation.py --mode multi_judge` and configurable via `TERTIARY_JUDGE_PROVIDER`; it is not configured for this run because no non-Yandex API key is accessible from this account, and the script gracefully runs with the two judges actually available.
>
> | Judge | Provider | Model URI | Faithful | Total | Rate | Wilson 95% CI |
> |---|---|---|---|---|---|---|
> | Primary | yandex | `gpt://{folder}/yandexgpt/latest` | 70 | 70 | 100.0% | [94.8%–100.0%] |
> | Secondary | yandex | `gpt://{folder}/yandexgpt-lite/latest` | 69 | 70 | 98.6% | [92.3%–99.7%] |
> | **Minimum (any-judge HALLUCINATION ⇒ HALLUCINATION)** | — | — | **69** | **70** | **98.6%** | **[92.3%–99.7%]** |
>
> Pairwise Cohen's κ on the n=70 intersection: **κ(primary, secondary) = 0.000 → "poor" by Landis & Koch (<0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 almost perfect)**. This value is mathematically degenerate, not a meaningful disagreement signal: the primary judge marks every case FAITHFUL, so its row marginal P(HALLUCINATION) = 0. With P(HALLUCINATION)=0 for one rater, expected agreement under marginal independence equals the observed agreement, forcing κ to 0 regardless of the actual disagreement. We report κ verbatim and flag this degeneracy explicitly rather than substitute a more flattering statistic — a more informative κ would require a judge that marks HALLUCINATION often enough to give the marginal a non-zero P(HALLUCINATION).
>
> The single disagreement is `cardio_40` (Tier 2 cardiology — congenital long QT syndrome following a resuscitated out-of-hospital cardiac arrest). The primary judge marked FAITHFUL; the secondary judge marked HALLUCINATION. The full retrieval-context-answer diagnostic dump for this case is in [`reports/judge_disagreement_inspection_2026-05-19.md`](judge_disagreement_inspection_2026-05-19.md) and the inter-judge analysis is in §5.3. Under the minimum-judge rule, the test-split tier breakdown is 70/70 FAITHFUL on every tier except T2 cardiology, which becomes 13/14 = 92.9% [Wilson 95% CI 68.5%–98.7%].
>
> Run cost reference: 70 test-split cases × 2 judges = 140 judge calls; total wall-clock 13.9 min on the Yandex API. Raw per-case verdicts are in [`reports/faithfulness_multijudge_raw_2026-05-19.csv`](faithfulness_multijudge_raw_2026-05-19.csv) and the markdown summary in [`reports/faithfulness_multijudge_2026-05-19.md`](faithfulness_multijudge_2026-05-19.md).

### 8.2 §7 Conclusion

> Headline metrics are reported on the 70-case held-out test split (§4.7), which excludes the 30 development cases used to tune K, L2 threshold, and chunk size. Faithfulness is now reported under the minimum-judge rule (a case counts FAITHFUL only if every configured judge agrees), not the single-judge rate. The multi-agent medical RAG system demonstrates strong performance across all three evaluation axes:
>
> - **Routing** achieves 100.0% accuracy (70/70) across all tiers on the held-out test split (§4.7), matching the full-set figure. The router demonstrates triage-like behaviour on cross-domain queries, consistently prioritising the presenting clinical urgency.
> - **Retrieval** achieves 88.6% Hit Rate (62/70) overall on the test split, with Cardiology at 82.9% (29/35) and Endocrinology at 94.3% (33/35). Recall is perfect on Tier 1 cardiology (100.0%, 13/13) and very high on Tier 1 endocrinology (91.7%, 11/12); performance drops on Tier 2 cardiology (78.6%, 11/14) and Tier 3 (out-of-scope, fallback-only behaviour), cleanly surfacing content gaps in the cardiology corpus that are independent of the tuning split.
> - **Faithfulness** reaches **98.6% (69/70) under the minimum-judge rule** on the held-out test split, with a **Wilson 95% CI lower bound of 92.3%** (§4.4). The primary YandexGPT judge marked every case FAITHFUL (100.0%); the secondary YandexGPT-Lite judge — given the identical strict prompt — disagreed on `cardio_40` (Tier 2 cardiology, congenital LQTS), applying a stricter token-grounding standard. The conservative 92.3% lower bound is the right number to quote when comparing this system to LLM-as-a-judge faithfulness results elsewhere; see §5.3 for the disagreement analysis and §6 Limitation 6 for the remaining same-vendor caveat.
>
> The hyperparameter grid search (K × L2 threshold, 30 combinations) was performed on the 30-case development split (§3.4) and confirmed K=5, L2 ≤ 1.2 as the optimal operating point, balancing retrieval completeness against context compactness for faithful generation. The chunk size optimization (400 words) and keyword-stripping strategy were both empirically validated and contributed measurably to system quality. The architecture is modular and ready for extension to additional medical specialties.

## 9. Smoke Test Output
```text
$ python -c "
from sklearn.metrics import cohen_kappa_score
yandex   = [True, True,  True,  False, True ]
secondary= [True, False, True,  False, True ]
tertiary = [True, True,  None,  False, True ]
kappa_ys = cohen_kappa_score(yandex, secondary)
assert -1 <= kappa_ys <= 1
both_faithful = [y and s and (t is True) for y, s, t in zip(yandex, secondary, tertiary) if t is not None]
min_rate = sum(both_faithful) / len(both_faithful)
print(f'mock yandex_rate=4/5  secondary_rate=3/5  tertiary_rate=3/4  min_judge_rate={min_rate}  κ_yandex_secondary={kappa_ys:.2f}')
assert min_rate == 2/4
print('Multi-judge smoke test passed')
"
mock yandex_rate=4/5  secondary_rate=3/5  tertiary_rate=3/4  min_judge_rate=0.5  κ_yandex_secondary=0.55
Multi-judge smoke test passed
```

## 10. Open Questions
- **Cross-vendor judge.** The current run is bounded by the fact that both judges are Yandex models; failure modes that look natural to all Yandex models are still undetectable. The next milestone is wiring a cross-vendor judge into `TERTIARY_JUDGE_PROVIDER` (the code path is already done — the blocker is API access).
- **κ degeneracy.** With one rater at 100% FAITHFUL, κ is structurally zero. If this state persists in future runs, consider reporting Gwet's AC1 alongside κ — AC1 is robust to one-sided marginals and would give a more informative agreement signal.
- **Single-disagreement statistical weight.** A single disagreement out of 70 cases is a high-variance estimate; rerunning the test split a few times (Yandex's stated determinism notwithstanding, `temperature=0.0` is not strict bitwise determinism) would let us estimate whether `cardio_40` is a stable rejection or a one-off.

## 11. Commit Message
`[eval] feat: add multi-judge faithfulness eval (primary yandexgpt + secondary yandexgpt-lite); minimum-judge rate becomes the §7 headline`
