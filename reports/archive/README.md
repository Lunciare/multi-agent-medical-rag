# reports/archive — Historical Stage Reports & Logs

Files moved here on 2026-05-22 to keep the top-level `reports/` listing
scannable. Each entry below was superseded by Stage-39 work
(`report_stage_indices_built.md`, `report_stage_dataset_extended.md`,
`report_stage_full_integration.md`, `report_stage_new_agents.md`) and
the canonical end-to-end document `report_final.md`. See
`reports/report_independent_audit.md` §7.3 for the inventory that
drove this cleanup.

Nothing here is intended for new readers — use the top-level reports
for current numbers. Files are kept on disk for traceability and for
the few historical citations in `multijudge_reconciliation.md`.

## Per-stage reports (29 files)

| File | Stage / focus |
|---|---|
| `report_stage_2.md`  | Stage 2 — initial keyword-stripping ablation (n=30 cardio) |
| `report_stage_3.md`  | Stage 3 — early chunk-size sweep |
| `report_stage_4.md`  | Stage 4 — routing baseline scaffolding |
| `report_stage_5.md`  | Stage 5 — first multi-judge faithfulness run |
| `report_stage_6.md`  | Stage 6 — cardio corpus-gap audit (cardio_23/25/35) |
| `report_stage_7.md`  | Stage 7 — single-threshold refusal gate (`L2_REJECT_MIN = 0.92`, 2-spec) |
| `report_stage_8.md`  | Stage 8 — refusal-gate dev-set tuning |
| `report_stage_9.md`  | Stage 9 — chunk-relevance LLM judge |
| `report_stage_10.md` | Stage 10 — adversarial routing v1 (n=32) |
| `report_stage_11.md` | Stage 11 — TF-IDF router baseline (cardio + endo only) |
| `report_stage_13.md` | Stage 13 — BM25 retriever comparison (2-spec) |
| `report_stage_14.md` | Stage 14 — 2×2 keyword-stripping × chunk-size ablation (cardio) |
| `report_stage_15.md` | Stage 15 — golden-dataset Fix 1 |
| `report_stage_16.md` | Stage 16 — interactive gold-source annotator |
| `report_stage_17.md` | Stage 17 — auto-annotator switch (`--auto`) |
| `report_stage_18.md` | Stage 18 — Recall@K denominator rework |
| `report_stage_19.md` | Stage 19 — MRR + bootstrap CI helpers |
| `report_stage_20.md` | Stage 20 — multijudge config slot (`SECONDARY_JUDGE_PROVIDER`) |
| `report_stage_21.md` | Stage 21 — Cohen's-κ degeneracy + Landis-Koch helper |
| `report_stage_22.md` | Stage 22 — failure-analysis report scaffolding |
| `report_stage_23.md` | Stage 23 — TERTIARY_JUDGE_PROVIDER slot |
| `report_stage_24.md` | Stage 24 — golden_dev/golden_test split (60/140) |
| `report_stage_25.md` | Stage 25 — original 32-case adversarial cohort |
| `report_stage_26.md` | Stage 26 — playwright UI smoke test |
| `report_stage_27.md` | Stage 27 — `_bootstrap_mean_ci` helper |
| `report_stage_28.md` | Stage 28 — generic single-class RefusalGate |
| `report_stage_29.md` | Stage 29 — retrieval regression snapshot |
| `report_stage_32.md` | Stage 32 — golden-dataset Fix 2 |
| `report_stage_33.md` | Stage 33 — pre-Stage-38 baseline freeze (last 2-spec snapshot) |

> Stage gaps (12, 30, 31): no separate report was produced; the work landed inside an adjacent stage.

## Older multi-judge faithfulness runs (4 files)

| File | Notes |
|---|---|
| `faithfulness_multijudge_2026-05-19.md` + `..._raw_2026-05-19.csv` | Stage-5 / pre-Stage-39 canonical multi-judge result (2-spec, n=58); cited by `multijudge_reconciliation.md`. |
| `faithfulness_multijudge_2026-05-20.md` + `..._raw_2026-05-20.csv` | Stage-21 re-run after the `_landis_koch` helper change; reproducibility witness. |

Current canonical multi-judge result lives at
`reports/faithfulness_multijudge_2026-05-21.{md,csv}` (n=140 test
split, post-Stage-39).

## Older routing-evaluation snapshots (13 files)

| File | Stage / what it was |
|---|---|
| `routing_evaluation_2026-05-18_15-52-08.md` | Stage 4 — first persisted routing eval (2-spec) |
| `routing_evaluation_2026-05-19_20-55-01.md` | Stage 5 / 6 |
| `routing_evaluation_2026-05-20_10-11-17.md` through `..._21-51-53.md` (9 files) | Stage 19 / 20 / 21 / 24 reruns (multiple iterations during dev/test split work) |
| `routing_evaluation_2026-05-21_11-23-03.md` | Stage 38 mid-iteration (pre-final dataset doubling) |

Current canonical routing eval:
`reports/routing_evaluation_2026-05-21_20-52-35.md` (LLM, n=140 test
split) and `reports/routing_evaluation_2026-05-21_20-53-59.md`
(adversarial, n=64).

## Older retrieval / judge-disagreement / log files (16 files)

| File | Notes |
|---|---|
| `retrieval_evaluation_2026-05-18_15-53-47.md` | Stage 4 — first persisted retrieval eval (2-spec) |
| `judge_disagreement_inspection_2026-05-19.md` | Stage 5 — `cardio_40` case write-up; superseded by `report_final.md` §5.3 |
| `ablation_cardiology_200_strip_2026-05-20.log`, `ablation_cardiology_2026-05-20.log`, `ablation_cardiology_400_keep_2026-05-20.log` | Stage 14 ablation raw stdout |
| `evaluation_with_ci_2026-05-20.log` | Stage 19 first run of `_bootstrap_mean_ci` helper |
| `generation_with_gate_2026-05-19.log`, `retrieval_with_gate_2026-05-19.log` | Stage 7 refusal-gate live runs |
| `multijudge_run_2026-05-19.log` | Stage 5 multi-judge stdout |
| `retrieval_grounded_all_2026-05-19.log`, `retrieval_grounded_test_2026-05-19.log` | Stage 18 Recall@K denominator rework |
| `retrieval_with_bm25_2026-05-20.log` | Stage 13 BM25 build raw stdout |
| `routing_post_refactor_dev_2026-05-20.log`, `routing_post_refactor_test_2026-05-20.log`, `routing_pre_refactor_dev_2026-05-20.log` | Stage 20 refactor diff |
| `test_set_results_2026-05-19.log` | Stage 24 first test-split eval |
