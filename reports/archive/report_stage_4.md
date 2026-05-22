# Stage 4 Report: Dev/Test Split for Held-Out Evaluation

**Date:** 2026-05-19

## 1. What Was Changed
- `multi-agent_system/tests/data/golden_dev.json` (new): 30-case development split, equal to the original tuning set `cardio_1..15` + `endo_1..15` (verified against `report_stage_2.md` §2.1).
- `multi-agent_system/tests/data/golden_test.json` (new): 70-case held-out test split — every remaining case in `golden_dataset.json`.
- `multi-agent_system/tests/evaluate_retrieval.py`, `evaluate_routing.py`, `evaluate_generation.py`, `evaluate_chunk_relevance.py`: added `--split {dev,test,all}` argument (default `test`). Each script now loads `golden_dev.json`, `golden_test.json`, or `golden_dataset.json` accordingly. The existing `--smoke-test` flag in `evaluate_retrieval.py` was preserved.
- `multi-agent_system/tests/tune_retrieval.py`, `tune_chunk_size.py`: hardcoded to read `golden_dev.json`. Both scripts accept a `--split` argument purely as a guard: any value other than `dev` exits with code 2 and the message `... is locked to --split dev ...`.
- `reports/test_set_results_2026-05-19.log` (new): full captured stdout of `evaluate_routing.py --split test`, `evaluate_retrieval.py --split test`, and `evaluate_generation.py --split test`.
- `reports/report_final.md`: inserted a split-explanation paragraph above §4.3 and a new §4.7 "Held-Out Test Set Results (n=70)" section with both retrieval and tiered-summary tables; rewrote §7 Conclusion to cite the §4.7 (test-split) numbers instead of the full-set §4.3/§4.6 numbers.

## 2. Hyperparameter Provenance

Prior hyperparameter selection (K, L2 threshold, chunk size) was performed exclusively on the 30-case development split (`golden_dev.json`) and was not informed by any case in the 70-case held-out test split.

### Dev-Set Hyperparameter Values (Unchanged)

| Parameter | Value | Source |
|---|---|---|
| `SIMILARITY_TOP_K` | **5** | Grid search on dev split (`tune_retrieval.py`, see [`hyperparameter_grid.csv`](hyperparameter_grid.csv)) |
| `MAX_L2_DISTANCE` | **1.2** | Grid search on dev split (`tune_retrieval.py`) |
| `CHUNK_SIZE_WORDS` | **400** | Grid search on dev split (`tune_chunk_size.py`) |
| `CHUNK_OVERLAP_WORDS` | 30 | Fixed |

The chosen operating point K=5, L2≤1.2 achieved 96.7% Hit Rate (29/30) on the dev split during the grid search; this matches the value recorded in `report_stage_2.md` §2.4 and is unchanged by the introduction of the dev/test split (no parameter was re-tuned).

## 3. Smoke Test Output
```text
$ cd multi-agent_system && python -c "
import json
dev = json.load(open('tests/data/golden_dev.json'))
test = json.load(open('tests/data/golden_test.json'))
assert len(dev) == 30, f'dev has {len(dev)} cases'
assert len(test) == 70, f'test has {len(test)} cases'
all_ids = {c['id'] for c in dev} | {c['id'] for c in test}
assert len(all_ids) == 100, 'dev and test overlap or are missing cases'
print('Split OK: 30 dev + 70 test, no overlap')
"
Split OK: 30 dev + 70 test, no overlap
```

## 4. Test-Set Headline Metrics (n=70, Wilson 95% CI)

| Metric | Successes / Total | Point Estimate | Wilson 95% CI |
|---|---|---|---|
| Routing Accuracy (overall) | 70 / 70 | **100.0%** | [94.8% – 100.0%] |
| Retrieval Hit Rate (overall) | 62 / 70 | **88.6%** | [79.0% – 94.1%] |
| Retrieval Hit Rate (Cardiology) | 29 / 35 | 82.9% | [67.3% – 91.9%] |
| Retrieval Hit Rate (Endocrinology) | 33 / 35 | 94.3% | [81.4% – 98.4%] |
| Faithfulness (overall) | 70 / 70 | **100.0%** | [94.8% – 100.0%] |

Per-tier CIs (matching the §4.7 table in `report_final.md`):

| Tier × Domain | Routing | Retrieval | Faithfulness |
|---|---|---|---|
| T1 Cardiology (n=13) | 13/13 = 100.0% [77.2–100%] | 13/13 = 100.0% [77.2–100%] | 13/13 = 100.0% [77.2–100%] |
| T1 Endocrinology (n=12) | 12/12 = 100.0% [75.8–100%] | 11/12 = 91.7% [64.6–98.5%] | 12/12 = 100.0% [75.8–100%] |
| T2 Cardiology (n=14) | 14/14 = 100.0% [78.5–100%] | 11/14 = 78.6% [52.4–92.4%] | 14/14 = 100.0% [78.5–100%] |
| T2 Endocrinology (n=16) | 16/16 = 100.0% [80.6–100%] | 15/16 = 93.8% [71.7–98.9%] | 16/16 = 100.0% [80.6–100%] |
| T3 Overall (n=15) | 15/15 = 100.0% [79.6–100%] | *fallback-only — see Limitations* | 15/15 = 100.0% [79.6–100%] |

Confidence intervals are computed via `statsmodels.stats.proportion.proportion_confint(..., method='wilson')`. Test-split tier composition: T1 = 25 (13 cardio + 12 endo), T2 = 30 (14 cardio + 16 endo), T3 = 15 (8 cardio + 7 endo).

## 5. report_final.md §4.7 — Verbatim Table Contents

The following two tables were inserted into `report_final.md` as §4.7. They are reproduced here verbatim.

### 4.7 Held-Out Test Set Results (n=70)

#### Retrieval Hit Rate (Test Split)

| Domain | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|
| Cardiology | 82.9% (29/35) | 49.7% | 25.7% | 6.3% |
| Endocrinology | 94.3% (33/35) | 69.7% | 20.0% | 4.6% |
| **Overall** | **88.6% (62/70)** | **59.7%** | **22.9%** | **5.4%** |

#### Summary of All Metrics (Test Split, n=70)

| Metric | T1 Cardiology (Core) | T1 Endocrinology (Core) | T2 Cardiology (Peripheral) | T2 Endocrinology (Peripheral) | T3 Overall (Out-of-Scope) |
|---|---|---|---|---|---|
| Routing Accuracy | 100.0% [77.2–100%] | 100.0% [75.8–100%] | 100.0% [78.5–100%] | 100.0% [80.6–100%] | 100.0% [79.6–100%] |
| Retrieval Hit Rate | 100.0% [77.2–100%] | 91.7% [64.6–98.5%] | 78.6% [52.4–92.4%] | 93.8% [71.7–98.9%] | *See Limitations* |
| Faithfulness | 100.0% [77.2–100%] | 100.0% [75.8–100%] | 100.0% [78.5–100%] | 100.0% [80.6–100%] | 100.0% [79.6–100%] |

## 6. Results Interpretation
a) **Generalisation to held-out cases.** Routing and faithfulness both hold at 100.0% (Wilson 95% CI lower bound 94.8%) on data the system was never tuned on. The dev-set tuning did not overfit those two axes.
b) **Retrieval gap is split-invariant.** Overall retrieval drops from the full-set 91.0% to 88.6% on test (as expected when the most textbook-aligned core cases are removed). The T2 cardiology Hit Rate (78.6%, 11/14) is identical to the figure reported in §4.3.1; the audited content gaps in `cardio_23`, `cardio_25`, and `cardio_35` are all in the test split, so the §4.3.1 audit is unaffected.
c) **Tier 3 behaviour is unchanged.** All 15 Tier 3 test cases retrieved 5 adjacent chunks; 0/15 triggered the "Insufficient evidence" fallback. This matches the architectural finding in §6 Limitation 8 and §5.2 and is not a tuning-driven artefact.
d) **Endocrinology T1 dip is small but real.** Test-split T1 endocrinology drops to 91.7% (11/12) vs. 100% on T1 cardiology. The Wilson 95% CI for T1 endo (64.6–98.5%) overlaps with T1 cardio's, so the difference is not separable at n=12.

## 7. Open Questions
- The 88.6% test-split Hit Rate sits below the 91.0% full-set figure but well within the 95% Wilson CI of the latter. Worth re-checking once the cardiology corpus expansion (§4.3.1 gaps) lands, to see whether the dev/test gap shrinks symmetrically.
- Tune scripts now reject `--split test/all`. If future stages want a one-off sanity check on the test split (e.g., to verify the tuned operating point still rules), the convention to use should be running `evaluate_retrieval.py --split test` (which already exists), not patching the tune scripts.

