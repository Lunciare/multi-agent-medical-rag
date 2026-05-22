# Routing Evaluation Report

**Date:** 2026-05-20 10:56:48

## Golden Dataset — Accuracy (Wilson 95% CI)

| Domain | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| cardiologist | 35 | 35 | 100.0% [90.1%–100.0%] |
| endocrinologist | 35 | 35 | 100.0% [90.1%–100.0%] |
| **Overall** | **70** | **70** | **100.0% [94.8%–100.0%]** |

## Golden Dataset — Routing Accuracy By Tier (Wilson 95% CI)

| Domain | Tier | Label | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 13 | 13 | 100.0% [77.2%–100.0%] |
| cardiologist | 2 | peripheral | 14 | 14 | 100.0% [78.5%–100.0%] |
| cardiologist | 3 | out_of_scope | 8 | 8 | 100.0% [67.6%–100.0%] |
| endocrinologist | 1 | core | 12 | 12 | 100.0% [75.8%–100.0%] |
| endocrinologist | 2 | peripheral | 16 | 16 | 100.0% [80.6%–100.0%] |
| endocrinologist | 3 | out_of_scope | 7 | 7 | 100.0% [64.6%–100.0%] |

## Golden Dataset — Per-Query Details

| ID | Expected | Predicted | Raw LLM Output | Result |
|---|---|---|---|---|
| cardio_16 | cardiologist | cardiologist | Cardiologist | V |
| cardio_17 | cardiologist | cardiologist | Cardiologist | V |
| cardio_18 | cardiologist | cardiologist | Cardiologist | V |
| cardio_19 | cardiologist | cardiologist | Cardiologist | V |
| cardio_20 | cardiologist | cardiologist | Cardiologist | V |
| cardio_21 | cardiologist | cardiologist | Cardiologist | V |
| cardio_22 | cardiologist | cardiologist | Cardiologist | V |
| cardio_23 | cardiologist | cardiologist | Cardiologist | V |
| cardio_24 | cardiologist | cardiologist | cardiologist | V |
| cardio_25 | cardiologist | cardiologist | Cardiologist | V |
| cardio_26 | cardiologist | cardiologist | Cardiologist | V |
| cardio_27 | cardiologist | cardiologist | Cardiologist | V |
| cardio_28 | cardiologist | cardiologist | Cardiologist | V |
| cardio_29 | cardiologist | cardiologist | Cardiologist | V |
| cardio_30 | cardiologist | cardiologist | Cardiologist | V |
| cardio_31 | cardiologist | cardiologist | Cardiologist | V |
| cardio_32 | cardiologist | cardiologist | Cardiologist | V |
| cardio_33 | cardiologist | cardiologist | Cardiologist | V |
| cardio_34 | cardiologist | cardiologist | Cardiologist | V |
| cardio_35 | cardiologist | cardiologist | Cardiologist | V |
| cardio_36 | cardiologist | cardiologist | Cardiologist | V |
| cardio_37 | cardiologist | cardiologist | Cardiologist | V |
| cardio_38 | cardiologist | cardiologist | Cardiologist | V |
| cardio_39 | cardiologist | cardiologist | Cardiologist | V |
| cardio_40 | cardiologist | cardiologist | Cardiologist | V |
| cardio_41 | cardiologist | cardiologist | Cardiologist | V |
| cardio_42 | cardiologist | cardiologist | Cardiologist | V |
| cardio_43 | cardiologist | cardiologist | Cardiologist | V |
| cardio_44 | cardiologist | cardiologist | Cardiologist | V |
| cardio_45 | cardiologist | cardiologist | Cardiologist | V |
| cardio_46 | cardiologist | cardiologist | Cardiologist | V |
| cardio_47 | cardiologist | cardiologist | Cardiologist | V |
| cardio_48 | cardiologist | cardiologist | Cardiologist | V |
| cardio_49 | cardiologist | cardiologist | Cardiologist | V |
| cardio_50 | cardiologist | cardiologist | Cardiologist | V |
| endo_16 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_17 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_18 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_19 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_20 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_21 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_22 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_23 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_24 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_25 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_26 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_27 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_28 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_29 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_30 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_31 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_32 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_33 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_34 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_35 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_36 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_37 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_38 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_39 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_40 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_41 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_42 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_43 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_44 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_45 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_46 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_47 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_48 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_49 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_50 | endocrinologist | endocrinologist | endocrinologist | V |

## Cross-Domain Ambiguous Cases

These queries intentionally span multiple medical domains. No single routing decision is considered "correct" — the table documents observed behaviour.

| ID | Label | Routed To | Valid Domains | In Valid? |
|---|---|---|---|---|
| ambig_1 | diabetic cardiomyopathy | cardiologist | cardiologist, endocrinologist | V |
| ambig_2 | thyroid-induced atrial fibrillation | endocrinologist | cardiologist, endocrinologist | V |
| ambig_3 | SGLT2 inhibitor cardioprotection in ACS | cardiologist | cardiologist, endocrinologist | V |
| ambig_4 | hyperaldosteronism with cardiac complications | endocrinologist | cardiologist, endocrinologist | V |
| ambig_5 | catecholamine-induced cardiomyopathy | endocrinologist | cardiologist, endocrinologist | V |
| ambig_6 | amiodarone-induced thyroid dysfunction | endocrinologist | cardiologist, endocrinologist | V |
| ambig_7 | metabolic syndrome with coronary artery disease | cardiologist | cardiologist, endocrinologist | V |
| ambig_8 | carcinoid heart disease | cardiologist | cardiologist, endocrinologist | V |
