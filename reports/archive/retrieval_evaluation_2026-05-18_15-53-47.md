# Retrieval Evaluation Report

**Date:** 2026-05-18 15:53:47
**K (SIMILARITY_TOP_K):** 5
**Random baseline seed:** 42

## Per-Domain Metrics (FAISS vs. Random Baseline)

| Domain | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|
| cardiologist | 86.0% | 56.4% | 36.0% | 10.4% |
| endocrinologist | 96.0% | 73.6% | 24.0% | 8.0% |
| **OVERALL** | **91.0%** | **65.0%** | **30.0%** | **9.2%** |

## By Tier

| Domain | Tier | Label | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|---|---|
| cardiologist | 1 | core | 100.0% | 74.1% | 44.4% | 13.3% |
| cardiologist | 2 | peripheral | 78.6% | 45.7% | 21.4% | 5.7% |
| cardiologist | 3 | out_of_scope | 55.6% | 20.0% | 33.3% | 8.9% |
| endocrinologist | 1 | core | 96.3% | 71.1% | 29.6% | 9.6% |
| endocrinologist | 2 | peripheral | 93.8% | 73.8% | 18.8% | 7.5% |
| endocrinologist | 3 | out_of_scope | 100.0% | 82.9% | 14.3% | 2.9% |

## Tier 3 (Out-of-Scope) — Fallback Behaviour

| Case ID | Chunks Retrieved | Flag |
|---|---|---|
| cardio_10 | 5 | ! ADJACENT CONTENT |
| cardio_28 | 5 | ! ADJACENT CONTENT |
| cardio_29 | 5 | ! ADJACENT CONTENT |
| cardio_30 | 5 | ! ADJACENT CONTENT |
| cardio_31 | 5 | ! ADJACENT CONTENT |
| cardio_32 | 5 | ! ADJACENT CONTENT |
| cardio_46 | 5 | ! ADJACENT CONTENT |
| cardio_47 | 5 | ! ADJACENT CONTENT |
| cardio_48 | 5 | ! ADJACENT CONTENT |
| endo_39 | 5 | ! ADJACENT CONTENT |
| endo_40 | 5 | ! ADJACENT CONTENT |
| endo_41 | 5 | ! ADJACENT CONTENT |
| endo_42 | 5 | ! ADJACENT CONTENT |
| endo_43 | 5 | ! ADJACENT CONTENT |
| endo_44 | 5 | ! ADJACENT CONTENT |
| endo_50 | 5 | ! ADJACENT CONTENT |
