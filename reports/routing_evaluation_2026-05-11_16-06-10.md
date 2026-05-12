# Routing Evaluation Report

**Date:** 2026-05-11 16:06:10

## Golden Dataset — Accuracy

| Domain | Correct | Total | Accuracy |
|---|---|---|---|
| cardiologist | 50 | 50 | 100.0% |
| endocrinologist | 49 | 49 | 100.0% |
| **Overall** | **99** | **99** | **100.0%** |

## Golden Dataset — Routing Accuracy By Tier

| Domain | Tier | Label | Correct | Total | Accuracy |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 27 | 27 | 100.0% |
| cardiologist | 2 | peripheral | 14 | 14 | 100.0% |
| cardiologist | 3 | out_of_scope | 9 | 9 | 100.0% |
| endocrinologist | 1 | core | 27 | 27 | 100.0% |
| endocrinologist | 2 | peripheral | 15 | 15 | 100.0% |
| endocrinologist | 3 | out_of_scope | 7 | 7 | 100.0% |

## Golden Dataset — Per-Query Details

| ID | Expected | Predicted | Raw LLM Output | Result |
|---|---|---|---|---|
| cardio_1 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_2 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_3 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_4 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_5 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_6 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_7 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_8 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_9 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_10 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_11 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_12 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_13 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_14 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_15 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_16 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_17 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_18 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_19 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_20 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_21 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_22 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_23 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_24 | cardiologist | cardiologist | cardiologist | ✅ |
| cardio_25 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_26 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_27 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_28 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_29 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_30 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_31 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_32 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_33 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_34 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_35 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_36 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_37 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_38 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_39 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_40 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_41 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_42 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_43 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_44 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_45 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_46 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_47 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_48 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_49 | cardiologist | cardiologist | Cardiologist | ✅ |
| cardio_50 | cardiologist | cardiologist | Cardiologist | ✅ |
| endo_1 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_2 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_3 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_4 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_5 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_6 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_7 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_8 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_9 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_10 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_11 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_12 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_13 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_14 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_15 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_16 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_17 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_18 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_19 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_20 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_21 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_22 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_23 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_24 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_25 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_26 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_27 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_28 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_29 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_30 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_31 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_32 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_33 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_34 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_36 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_37 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_38 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_39 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_40 | endocrinologist | endocrinologist | Endocrinologist | ✅ |
| endo_41 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_42 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_43 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_44 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_45 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_46 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_47 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_48 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_49 | endocrinologist | endocrinologist | endocrinologist | ✅ |
| endo_50 | endocrinologist | endocrinologist | endocrinologist | ✅ |

## Cross-Domain Ambiguous Cases

These queries intentionally span multiple medical domains. No single routing decision is considered "correct" — the table documents observed behaviour.

| ID | Label | Routed To | Valid Domains | In Valid? |
|---|---|---|---|---|
| ambig_1 | diabetic cardiomyopathy | cardiologist | cardiologist, endocrinologist | ✅ |
| ambig_2 | thyroid-induced atrial fibrillation | endocrinologist | cardiologist, endocrinologist | ✅ |
| ambig_3 | SGLT2 inhibitor cardioprotection in ACS | cardiologist | cardiologist, endocrinologist | ✅ |
| ambig_4 | hyperaldosteronism with cardiac complications | endocrinologist | cardiologist, endocrinologist | ✅ |
| ambig_5 | catecholamine-induced cardiomyopathy | endocrinologist | cardiologist, endocrinologist | ✅ |
| ambig_6 | amiodarone-induced thyroid dysfunction | endocrinologist | cardiologist, endocrinologist | ✅ |
| ambig_7 | metabolic syndrome with coronary artery disease | cardiologist | cardiologist, endocrinologist | ✅ |
| ambig_8 | carcinoid heart disease | cardiologist | cardiologist, endocrinologist | ✅ |
