# Routing Evaluation Report

**Date:** 2026-05-20 10:11:17

## Golden Dataset — Accuracy

| Domain | Correct | Total | Accuracy |
|---|---|---|---|
| cardiologist | 15 | 15 | 100.0% |
| endocrinologist | 15 | 15 | 100.0% |
| **Overall** | **30** | **30** | **100.0%** |

## Golden Dataset — Routing Accuracy By Tier

| Domain | Tier | Label | Correct | Total | Accuracy |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 14 | 14 | 100.0% |
| cardiologist | 3 | out_of_scope | 1 | 1 | 100.0% |
| endocrinologist | 1 | core | 15 | 15 | 100.0% |

## Golden Dataset — Per-Query Details

| ID | Expected | Predicted | Raw LLM Output | Result |
|---|---|---|---|---|
| cardio_1 | cardiologist | cardiologist | Cardiologist | V |
| cardio_2 | cardiologist | cardiologist | Cardiologist | V |
| cardio_3 | cardiologist | cardiologist | Cardiologist | V |
| cardio_4 | cardiologist | cardiologist | Cardiologist | V |
| cardio_5 | cardiologist | cardiologist | Cardiologist | V |
| cardio_6 | cardiologist | cardiologist | Cardiologist | V |
| cardio_7 | cardiologist | cardiologist | Cardiologist | V |
| cardio_8 | cardiologist | cardiologist | Cardiologist | V |
| cardio_9 | cardiologist | cardiologist | Cardiologist | V |
| cardio_10 | cardiologist | cardiologist | Cardiologist | V |
| cardio_11 | cardiologist | cardiologist | Cardiologist | V |
| cardio_12 | cardiologist | cardiologist | Cardiologist | V |
| cardio_13 | cardiologist | cardiologist | Cardiologist | V |
| cardio_14 | cardiologist | cardiologist | Cardiologist | V |
| cardio_15 | cardiologist | cardiologist | Cardiologist | V |
| endo_1 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_2 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_3 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_4 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_5 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_6 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_7 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_8 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_9 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_10 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_11 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_12 | endocrinologist | endocrinologist | Endocrinologist | V |
| endo_13 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_14 | endocrinologist | endocrinologist | endocrinologist | V |
| endo_15 | endocrinologist | endocrinologist | Endocrinologist | V |

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
