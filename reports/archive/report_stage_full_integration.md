# Stage: Full Four-Specialty Integration

(Scope: generalise the eval infrastructure to all four specialists, re-tune
the refusal gate on the 60-case dev set, run the full 4-specialty
evaluation suite — routing, retrieval, chunk relevance, multi-judge
faithfulness — and rewrite `report_final.md` §4 / §6 L3 / §7 to reflect the
new numbers. Inherits from `report_stage_indices_built.md` (Prompt 1,
indices on disk) and `report_stage_dataset_extended.md` (Prompt 2, 200-case
dataset). Verifies cardiology + endocrinology numbers remain byte-identical
to the 2-specialty baseline.)

## 1. Eval-script generalisation diffs (Part 1)

Six scripts had hardcoded 2-specialty `("cardiologist", "endocrinologist")`
tuples or `if expected == "cardiologist": ... else: ...` agent lookups.
All were replaced with registry-driven equivalents. No script behaviour
changed for the existing cardio + endo cases (verified by §7 regression
check); the same code path now iterates four agents instead of two.

### evaluate_routing.py

- `ALLOWED_SPECIALISTS = {"cardiologist", "endocrinologist"}`
  → `ALLOWED_SPECIALISTS = set(AGENT_REGISTRY.keys())` (now 4-element)
- The 2-specialty routing prompt was rewritten as a registry-driven
  `_build_routing_system_prompt()` that inlines each agent's
  `domain_scope` (mirrors `orchestrator._routing_system_prompt`).
  Output token length: 1,225 chars / ~310 tokens — well within YandexGPT's
  effective system-prompt window.
- Per-tier loop `for domain in ("cardiologist", "endocrinologist"):`
  → `for domain in sorted(AGENT_REGISTRY.keys()):` (two sites — table
  print + markdown emit).

### evaluate_routing_baseline.py

- Added `ENDO_KEYWORDS`, `GASTRO_KEYWORDS`, `INFECT_KEYWORDS` sets
  (existing `CARDIO_KEYWORDS` unchanged).
- `keyword_route()` reframed from binary "cardiology vs rest" to
  "highest hit count wins, registry-order tie-break". The Stage-1
  endocrinology fallback is preserved when no keyword set matches
  (this is what produces the 45.7% infectiology accuracy reported in
  §4.1 — many T1 infect cases use endocrine-flavoured surface vocab
  that triggers the fallback).
- Table-printer functions rewritten to span 4 columns instead of 2.

### evaluate_retrieval.py

- Smoke-test dataset assertions updated for the 200-case world:
  `expected_total = 200`, `expected_per_specialty = 50`,
  `expected_tiers` now includes per-specialty tier counts for all four.
- Per-domain accumulators (`domain_hits`, `domain_precision_sum`,
  `domain_total`, `domain_pool`, `domain_recall_sum`,
  `domain_mrr_per_case`) all switched from hardcoded
  `{"cardiologist": ..., "endocrinologist": ...}` literals to
  registry-driven dict comprehensions `{k: 0 for k in AGENT_REGISTRY}`.
- FAISS / BM25 / Random / Oracle pooled grid `pooled_hits` /
  `pooled_gold` / `mrr_sum` widened from cardio + endo to all four
  specialties × T1 / T2.
- `print_sources()` agent lookup generalised from the 2-specialty
  if/else to `orchestrator.agents.get(case["expected_specialist"])`.

### evaluate_generation.py

- `domain_faithful` / `domain_total` dicts → registry-driven.
- Both `_evaluate_multi_judge()` and `_evaluate_yandex_only()` agent
  lookups simplified from the 2-specialty if/else to
  `orchestrator.agents.get(expected_agent)`.

### evaluate_chunk_relevance.py

- Same pattern: registry-driven dicts, single-call agent lookup,
  4-specialty per-domain printer loop.

### tune_refusal_gate.py

- `_collect_min_dists()` agent lookup generalised.
- The two-specialty Signal-B `cardio_stats` / `endo_stats` block
  → registry-driven `corpus_stats = {key: load_or_compute_corpus_dist_stats(...)
  for key in AGENT_REGISTRY}`. The per-specialty corpus-distribution
  cache files (`data/processed/{specialty}/corpus_dist_stats.json`)
  now exist for all four specialties; gastro and infect were auto-created
  by `RefusalGate.from_vectorstore()` during Prompt 2's spot check.
- Signal-B `thr_by_domain` dict computed by dict comprehension over
  every registry entry.

### annotate_gold_sources.py (Prompt 2 had to fix this too)

The hardcoded if/else lookup was replaced with
`orchestrator.agents[case["expected_specialist"]]` in Prompt 2. This
stage inherits that fix.

## 2. Refusal-gate re-tune (Part 2)

Re-tune ran `python tests/tune_refusal_gate.py` on the new 60-case dev
split (3 T3 cases vs the prior 30-case dev set's 1 T3 case). Full grid
preserved at `reports/refusal_gate_grid.csv`. **Both targets unmet at
4-specialty scale** — the dev-only `≥80% T3 recall AND ≤5% T1/T2 FP`
constraint has no feasible row.

Per-specialty corpus-distribution stats (μ, σ over a random sample of
1,000 in-corpus chunks per specialty):

| Specialty | μ (mean nearest-neighbour L2) | σ |
|---|---:|---:|
| cardiologist | 0.8738 | 0.1094 |
| endocrinologist | 0.8940 | 0.0961 |
| gastroenterologist | 0.8798 | 0.0911 |
| infectionist | 0.8939 | 0.0962 |

Cardiology's μ is the lowest (cleanest separation between in-corpus and
out-of-corpus); infectiology and endocrinology cluster at the top
(weakest separation).

Headline trade-off:

| `L2_REJECT_MIN` | Test T3 recall | Test T1/T2 FP rate | Chosen at |
|---:|---|---|---|
| 0.920 | 89.7% (26/29) | 71.2% (79/111) | Stage 7 (2-specialty) |
| **1.020** | **20.7% (6/29)** | **17.1% (19/111)** | **Stage 39 (this stage, best-F1)** |
| 1.050 | 10.3% (3/29) | 0.0% (0/111) | (not chosen — T3 too low) |

The tuner's selection-rule fall-through to "best F1 row" produced
`L2_REJECT_MIN = 1.020`. The trade-off has shifted dramatically: the gate
now catches a fifth of T3 cases (vs prior 90%) but falsely refuses only
a sixth of T1/T2 (vs prior 71%). Both targets unmet; a two-stage gate
(L2 pre-filter + LLM-as-classifier confirmer) is the proposed fix
documented in `report_final.md` §6 L8.

`settings.py` was updated by the tuner:

```
REFUSAL_GATE_SIGNAL = 'A'
L2_REJECT_MIN = 1.020   # was 0.920 at Stage 7
```

The `corpus_dist_stats.json` cache files for cardiology and endocrinology
were re-computed by the tuner because the orchestrator construction
now eagerly loads all four agents (re-computation is idempotent — same
μ, σ to 4 decimal places as the Stage-31 files).

## 3. Headline numbers for every Part-3 run, with deltas vs 2-specialty baseline

### 3a. Routing on test split (140 cases, ~5 min, ~140 LLM calls)

```
Routing Evaluation — Golden Dataset (Wilson 95% CI)
  Domain                Correct    Total  Accuracy [Wilson 95% CI]
  cardiologist               35       35  100.0% [90.1%–100.0%]
  endocrinologist            34       35  97.1% [85.5%–99.5%]
  gastroenterologist         31       35  88.6% [74.0%–95.5%]
  infectionist               34       35  97.1% [85.5%–99.5%]
  OVERALL                   134      140  95.7% [91.0%–98.0%]
```

**Δ vs Stage-31 (n=70 test split, 2-specialty):**
- Cardio: 100% (35/35) → 100% (35/35) — IDENTICAL ✓
- Endo: 100% (35/35) → 97.1% (34/35) — 1-case drift (endo_48 → infectionist; not a regression because infectionist didn't exist at Stage 31)
- Gastro / infect: NEW columns at 88.6% / 97.1%

The 6 LLM misses are all defensible cross-specialty ambiguities; full
list in §4.1 of `report_final.md`.

### 3b. Routing on adversarial split (64 cases)

```
Adversarial Routing — Per-Category Accuracy
  Category                            Correct  Total  Accuracy [Wilson 95% CI]
  dominant_pathology_mismatch              16     16  100.0% [80.6%–100.0%]
  misspelled                               16     16  100.0% [80.6%–100.0%]
  non_english                              16     16  100.0% [80.6%–100.0%]
  symptom_only_ambiguous                   15     16  93.8%  [71.7%–98.9%]
  OVERALL                                  63     64  98.4%  [91.7%–99.7%]
```

**Δ vs Stage-25 (n=32 adversarial, 2-specialty):** every category was 8/8
= 100% at Stage 25; doubling to n=16 tightens the Wilson lower bound from
67.6% to 80.6%, and `symptom_only_ambiguous` drops 1 case (gastro/infect
symptom-only cases are harder than the original cardio/endo ones).

### 3c. Retrieval on test split (140 cases, no LLM, ~1 min)

```
Retriever Comparison — Recall@5 and MRR@5
  Overall T1+T2 faiss      162    285  56.8% [51.0%–62.5%]   0.704 [0.631–0.777]
  Overall T1+T2 bm25        98    285  34.4% [29.1%–40.1%]   0.519 [0.434–0.602]
  Overall T1+T2 random       8    285   2.8% [1.4%–5.4%]     0.037 [0.011–0.071]
  Overall T1+T2 oracle     285    285 100.0% [98.7%–100.0%]   1.000 [1.000–1.000]
```

**Δ vs Stage-13 (2-specialty test, n=70):**
- Overall FAISS Recall@5: 56.2% (86/153) → 56.8% (162/285) — within noise.
- Cardio T1: 59.0% (23/39) → 59.0% (23/39) IDENTICAL ✓
- Cardio T2: 54.1% (20/37) → 54.1% (20/37) IDENTICAL ✓
- Endo T1: 60.6% (20/33) → 60.6% (20/33) IDENTICAL ✓
- Endo T2: 52.3% (23/44) → 52.3% (23/44) IDENTICAL ✓
- Gastro T1 / T2: 57.6% (19/33) / 60.0% (21/35) (NEW)
- Infect T1 / T2: 50.0% (16/32) / 62.5% (20/32) (NEW)

FAISS still beats BM25 by 22.4 pp overall (was 26 pp on 2-specialty);
narrows on the new corpora's T1 (gastro 12.1 pp, infect 18.8 pp) where
exact-entity-name retrieval helps.

### 3d. Chunk-relevance on test split (140 LLM judge calls, ~10 min)

```
Chunk Relevancy Evaluation Results
  Domain               Sufficient  Total  Relevancy [Wilson 95% CI]
  cardiologist                 35     35  100.0% [90.1%–100.0%]
  endocrinologist              34     35  97.1%  [85.5%–99.5%]
  gastroenterologist           35     35  100.0% [90.1%–100.0%]
  infectionist                 35     35  100.0% [90.1%–100.0%]
  OVERALL                     139    140  99.3%  [96.1%–99.9%]
```

The 1 miss is the same case (`endo_25`) that has surfaced in prior
stages as a borderline relevance call.

### 3e. Multi-judge faithfulness on test split (266 judge calls, 15.2 min)

```
Per-Judge Faithfulness (test split, Wilson 95% CI)
  Judge            Faithful   Total Judged   Rate    Wilson 95% CI
  yandex_primary   132        132            100.0%  [97.2%–100.0%]
  secondary        131        132             99.2%  [95.8%–99.9%]

Minimum-Judge (all judges = FAITHFUL): 131/132 = 99.2% [95.8%–99.9%]
Pairwise Cohen's κ: κ(primary, secondary) = 0.000 — degenerate (one marginal = 0)

Disagreement cases: 1
  cardio_40  T2  cardiologist  primary=FAITHFUL  secondary=HALLUCINATION

Tier 3 Fallback: 7 / 29 (3 gastro T3 + 4 infect T3) — excluded from judge totals
```

**Δ vs Stage-31 (n=70):**
- Single disagreement is **still `cardio_40`** — the 70 new gastro/infect
  test cases introduced no new disagreements ✓ (regression check for
  faithfulness passes).
- Min-judge rate rose from 98.6% (69/70, Stage 31) → 99.2% (131/132,
  Stage 39) because the denominator doubled while the disagreement
  count stayed at 1. The Wilson lower bound tightened from 92.3% → 95.8%.

## 4. §4 / §7 / §6 L3 diffs in report_final.md

`reports/report_final.md` was rewritten end-to-end across §4.1–§4.10,
§5.1, §5.2, §6 L1–L11, and §7. The full diff is in the working-tree
state; the highlights are:

- §4 intro: golden dataset description bumped from "100 cases across
  three tiers" to "200 cases across three tiers and four specialties".
- §4.1 routing baseline table: 3 → 5 columns (added Gastroenterology,
  Infectiology, kept Overall). LLM Router headline: 100% → 95.7%.
- §4.2 ambiguous cases table: 8 → 14 rows; LLM stays in `valid_domains`
  on 13/14 (the exception is ambig_8 routed to gastroenterologist —
  flagged as dataset-curation follow-up).
- §4.2.1 adversarial: 32 → 64 cases; per-category Wilson lower bounds
  tighten from 67.6% to 80.6%; overall lower bound 89.3% → 91.7%.
- §4.3 retrieval: 2 → 4 specialty rows. Recall@5 overall 59.2% (142/240)
  → 56.8% (162/285) (denominator scales because each specialty
  contributes ~70 gold-doc trials). Tier-3 refusal column moves from
  the legacy zero-chunk metric to the Stage-7 numeric gate.
- §4.3.1 corpus-coverage audit: extended from 3 cardio cases to 8
  cases (3 cardio + 1 endo + 3 gastro + 3 infect) with concrete
  source-material remediation suggestions per case.
- §4.3.2 retriever comparison: 4 → 8 rows (4 specialties × T1/T2);
  FAISS-vs-BM25 gap is 22.4 pp overall (vs 26 pp on 2-specialty),
  narrower on the new specialties' T1 where exact-entity-name retrieval
  helps.
- §4.4 faithfulness: judges-table denominator 70 → 132; min-judge rate
  98.6% (69/70) → 99.2% (131/132). Per-domain table extended to 4
  specialties; only T2 cardiology is below 100% (cardio_40 disagreement,
  same as Stage 31). New paragraph documenting the 7 Tier-3 fallback
  exclusions.
- §4.5 refusal gate: **complete rewrite**. New per-specialty L2-range
  table replaces the Stage-30 architectural-framing paragraph. New
  target-check table shows BOTH targets unmet (was: T3 target met,
  FP target missed). New trade-off-curve table compares 5 thresholds
  on the 4-specialty test split. The 0.92 → 1.020 threshold change
  is documented with the explicit trade-off (69 pp T3-recall loss,
  54 pp FP-rate gain).
- §4.6 retrieval-regression: 10 → 20 queries (Prompt 1 already did this).
- §4.7 summary-of-metrics: table widened from 5 columns (T1 cardio +
  T1 endo + T2 cardio + T2 endo + T3 overall) to 9 columns
  (per-specialty × per-tier). Faithfulness row now references the
  test-split min-judge numbers.
- §4.8 held-out-test-results: n=70 → n=140; new per-specialty rows
  in every sub-table; new chunk-relevance row added.
- §4.9 PubMedQA: paragraph appended noting the Stage-27 calibration
  was cardio-only and gastro/infect re-probing is a follow-up.
- **§4.10 NEW**: Regression check section explicitly listing every
  cardio + endo cell that retained its 2-specialty value (Recall@5 T1/T2
  for both, BM25 T1/T2, cardio_40 single-disagreement identity).
- §5.1: overall Hit Rate 91.0% → KeywordHitRate 85.7% on test; the
  Recall@5-vs-KeywordHitRate framing preserved.
- §5.2: refusal-gate trade-off narrative rewritten to discuss both the
  Stage-7 0.92 setpoint and the Stage-39 1.020 setpoint.
- §6 L1: dataset size 100 → 200.
- §6 L2: corpus-gap list expanded from 3 cardio cases to 8 cases
  across all 4 specialties.
- **§6 L3: completely rewritten** from "two FAISS indices pending" to
  "four specialists evaluated end-to-end" with the Stage-39 generalisation
  documented.
- §6 L4: token-limit paragraph extended with the Stage-36 PDF-artifact
  filter (`_MAX_MEAN_WORD_LEN_CHARS = 15`).
- §6 L8: refusal-gate limitation upgraded from "trades FP for recall"
  to "both targets unmet at 4-specialty scale; two-stage gate is urgent".
- **§6 L9, L10, L11 NEW**: schema-drift in `ambiguous_cases.json` /
  Stage-31 multijudge reconciliation still on n=70 / TF-IDF baseline
  needs re-training on 60-case dev.
- §7 Conclusion: 4 headline bullets rewritten with n=140 test numbers;
  drops "two FAISS indices pending" cross-reference. The final
  paragraph now states "all four specialists evaluated end-to-end".

## 5. README evaluation table before/after

The README's Evaluation Results table grew from 5 columns (T1 cardio,
T1 endo, T2 cardio, T2 endo, T3 overall + Overall) to a 6-column
per-specialty layout. Numbers replaced; structure changed to:

- Tier composition table: 4 specialty columns + Total
- Evaluation metrics table: 4 specialty columns + Overall, 6 metric
  rows (Routing, Recall@5 T1, Recall@5 T2, Faithfulness, T3 Refusal
  Rate, T1/T2 FP Rate)

Footnote updated to reference the n=140 test split and the Stage-39
re-tuned `L2_REJECT_MIN = 1.020`. Limitations bullets refreshed —
"FAISS pending" wording removed (replaced with the Stage-39 "four
specialists evaluated end-to-end" framing).

## 6. New pytest count + test list

```
$ python -m pytest tests/ -q
51 passed, 1 skipped, 8 warnings in 6.28s
```

**51 passed = 41 pre-Stage-39 baseline + 10 new tests in tests/test_4_specialists.py.**

New test classes:

```
TestFourSpecialistConstruction
  test_orchestrator_constructs_all_4_specialists           PASSED
  test_allowed_specialists_includes_all_4                  PASSED

TestFourSpecialistRouting
  test_route_returns_gastroenterologist                    PASSED
  test_route_returns_infectionist                          PASSED

TestFourSpecialistAnswerFlow
  test_gastro_agent_answer_with_mock                       PASSED
  test_infect_agent_answer_with_mock                       PASSED

TestFourSpecialistRefusalGate
  test_refusal_gate_engages_for_cardiologist               PASSED
  test_refusal_gate_engages_for_endocrinologist            PASSED
  test_refusal_gate_engages_for_gastroenterologist         PASSED
  test_refusal_gate_engages_for_infectionist               PASSED
```

The 1 skipped test is the pre-existing Playwright browser-binary skip;
the 8 warnings are upstream-dep deprecations (Gradio + Pandas), all
pre-existing.

## 7. Regression check output (Part 7)

Cardio and endo numbers post-Stage-39 vs Stage-31 baseline, on the 35
cardio + 35 endo test cases that are present in both the n=70 (Stage 31)
and n=140 (Stage 39) test splits:

| Metric | Cardiology | Endocrinology | Source |
|---|---|---|---|
| Routing accuracy (all tiers) | 100% (35/35) ✓ IDENTICAL | 97.1% (34/35) ✗ 1-case shift | §4.1 |
| Recall@5 T1 (pooled gold) | 59.0% (23/39) ✓ IDENTICAL | 60.6% (20/33) ✓ IDENTICAL | §4.3.2 |
| Recall@5 T2 (pooled gold) | 54.1% (20/37) ✓ IDENTICAL | 52.3% (23/44) ✓ IDENTICAL | §4.3.2 |
| BM25 Recall@5 T1 | 25.6% (10/39) ✓ IDENTICAL | 15.2% (5/33) ✓ IDENTICAL | §4.3.2 |
| BM25 Recall@5 T2 | 43.2% (16/37) ✓ IDENTICAL | 34.1% (15/44) ✓ IDENTICAL | §4.3.2 |
| Multi-judge disagreements | {cardio_40} ✓ IDENTICAL | {} ✓ IDENTICAL | §4.4 |
| Minimum-judge faithfulness | 92.9% (13/14) on T2 — IDENTICAL | 100% across all tiers — IDENTICAL | §4.4 |
| §4.3.1 corpus-gap cases | cardio_23, cardio_25, cardio_35 ✓ | endo_46 ✓ | §4.3.1 |

**Endo 34/35 vs prior 35/35 is the single deviation**, and it is not a
regression: `endo_48` (SGLT2-inhibitor UTI/DKA) routed to infectionist
in the 4-specialty system. Infectionist did not exist in the 2-specialty
system; the routing decision is clinically plausible.

Per-cell verification confirms the Stage-39 eval refactor preserved
per-specialty isolation. No cardio or endo number changed because of
the registry-driven loop reformulation.

## 8. Open follow-ups (deferred to future stages)

1. **TF-IDF baseline retrain.** `tests/data/tfidf_router.pkl` was trained
   pre-Stage-39 (cardio+endo only). Retrain on the new 60-case 4-specialty
   dev split via `tests/train_tfidf_router.py`. Cost: one LR refit (<1 min),
   no API calls. Should bring gastro/infect columns of §4.1 above 0%.

2. **Two-stage refusal gate.** Per §4.5 + §6 L8, the single-threshold
   numeric gate cannot meet both targets at 4-specialty scale. Proposed
   design: numeric pre-filter for clear cases (L2 < ~0.85 → pass; L2 >
   ~1.10 → refuse); LLM-as-classifier confirmer for the overlap zone.
   Cost estimate: ~20% of queries fall in the overlap zone × 1 cheap
   LLM call each = ~28 calls per 140-case eval, marginal cost.

3. **Stage-31 multijudge reconciliation update.** Re-run the
   reconciliation note for the 4-specialty test split. Cost: same as
   Part 3e (~280 judge calls, ~15 min, ~1500 ₽). Goal: verify the
   single-disagreement-case identity stays at cardio_40 across multiple
   runs at temperature=0 on the new larger denominator.

4. **PubMedQA gastro + infect slices.** §4.9 PubMedQA is cardio-only;
   add per-specialty PubMedQA slices using analogous keyword filters
   (gastro: `gastr|hepatic|colon|liver|crohn|colitis|hepatitis|cirrhosis|pancrea`;
   infect: `infect|antibiotic|antiviral|sepsis|HIV|tuberculosis|pneumonia|malaria`).
   The Jaccard threshold may need per-specialty calibration because
   corpus register varies. Cost: no LLM, just retrieval + Jaccard.

5. **Stage-14 chunk-size ablation on gastro + infect.** Stage 14 was
   cardio-only (2×2 grid over chunk size × keyword stripping). Repeating
   it on the new corpora would tell us whether 400-word chunks are still
   optimal; needs a fresh FAISS rebuild per ablation cell (~50 min each
   on the new corpora), ~4 hours total.

6. **`ambiguous_cases.json` schema unification.** Cases 1–8 use only
   `domains`; cases 9–14 use both `valid_domains` and `domains`. Adversarial
   `symptom_only_ambiguous` cases use `valid_domains` only. Pick one
   convention and migrate. Cost: no eval re-run; just a one-line script.

7. **Refusal-gate logic in `report_final.md` §6 L8 vs §4.5.** §4.5
   currently describes both the Stage-7 baseline and the Stage-39 re-tune.
   §6 L8 mostly retains the Stage-7 framing. A future-stage cleanup
   could collapse §6 L8 into a pointer to §4.5; out of scope here.