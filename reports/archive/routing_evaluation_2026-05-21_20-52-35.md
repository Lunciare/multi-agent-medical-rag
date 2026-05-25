# Routing Evaluation Report

**Date:** 2026-05-21 20:52:35

## Golden Dataset — Accuracy (Wilson 95% CI)

| Domain | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| cardiologist | 35 | 35 | 100.0% [90.1%–100.0%] |
| endocrinologist | 34 | 35 | 97.1% [85.5%–99.5%] |
| gastroenterologist | 31 | 35 | 88.6% [74.0%–95.5%] |
| infectionist | 34 | 35 | 97.1% [85.5%–99.5%] |
| **Overall** | **134** | **140** | **95.7% [91.0%–98.0%]** |

## Golden Dataset — Routing Accuracy By Tier (Wilson 95% CI)

| Domain | Tier | Label | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 13 | 13 | 100.0% [77.2%–100.0%] |
| cardiologist | 2 | peripheral | 14 | 14 | 100.0% [78.5%–100.0%] |
| cardiologist | 3 | out_of_scope | 8 | 8 | 100.0% [67.6%–100.0%] |
| endocrinologist | 1 | core | 12 | 12 | 100.0% [75.8%–100.0%] |
| endocrinologist | 2 | peripheral | 15 | 16 | 93.8% [71.7%–98.9%] |
| endocrinologist | 3 | out_of_scope | 7 | 7 | 100.0% [64.6%–100.0%] |
| gastroenterologist | 1 | core | 11 | 13 | 84.6% [57.8%–95.7%] |
| gastroenterologist | 2 | peripheral | 13 | 15 | 86.7% [62.1%–96.3%] |
| gastroenterologist | 3 | out_of_scope | 7 | 7 | 100.0% [64.6%–100.0%] |
| infectionist | 1 | core | 12 | 13 | 92.3% [66.7%–98.6%] |
| infectionist | 2 | peripheral | 15 | 15 | 100.0% [79.6%–100.0%] |
| infectionist | 3 | out_of_scope | 7 | 7 | 100.0% [64.6%–100.0%] |

## Golden Dataset — Per-Query Details

| ID | Expected | Predicted | Raw LLM Output | Result |
|---|---|---|---|---|
| cardio_16 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_17 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_18 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_19 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_20 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_21 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_22 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_23 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_24 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_25 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_26 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_27 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_28 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_29 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_30 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_31 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_32 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_33 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_34 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_35 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_36 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_37 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_38 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_39 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_40 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_41 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_42 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_43 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_44 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_45 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_46 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_47 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_48 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_49 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| cardio_50 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| endo_16 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_17 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_18 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_19 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_20 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_21 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_22 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_23 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_24 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_25 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_26 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_27 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_28 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_29 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_30 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_31 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_32 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_33 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_34 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_35 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_36 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_37 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_38 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_39 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_40 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_41 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_42 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_43 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_44 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_45 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_46 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_47 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_48 | endocrinologist | infectionist | {"specialist": "infectionist"} | X |
| endo_49 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| endo_50 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| gastro_16 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_17 | gastroenterologist | endocrinologist | {"specialist": "endocrinologist"} | X |
| gastro_18 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_19 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_20 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_21 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_22 | gastroenterologist | endocrinologist | {"specialist": "endocrinologist"} | X |
| gastro_23 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_24 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_25 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_26 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_27 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_28 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_29 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_30 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_31 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_32 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_33 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_34 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_35 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_36 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_37 | gastroenterologist | infectionist | {"specialist": "infectionist"} | X |
| gastro_38 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_39 | gastroenterologist | endocrinologist | {"specialist": "endocrinologist"} | X |
| gastro_40 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_41 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_42 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_43 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_44 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_45 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_46 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_47 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_48 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_49 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| gastro_50 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| infect_16 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_17 | infectionist | gastroenterologist | {"specialist": "gastroenterologist"} | X |
| infect_18 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_19 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_20 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_21 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_22 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_23 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_24 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_25 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_26 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_27 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_28 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_29 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_30 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_31 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_32 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_33 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_34 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_35 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_36 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_37 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_38 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_39 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_40 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_41 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_42 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_43 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_44 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_45 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_46 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_47 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_48 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_49 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| infect_50 | infectionist | infectionist | {"specialist": "infectionist"} | V |

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
| ambig_8 | carcinoid heart disease | gastroenterologist | cardiologist, endocrinologist | ? |
| ambig_9 | H. pylori peptic ulcer with iron deficiency | gastroenterologist | gastroenterologist, endocrinologist | V |
| ambig_10 | sepsis with new-onset hyperglycaemia | endocrinologist | infectionist, endocrinologist | V |
| ambig_11 | HIV with cardiac complications | cardiologist | infectionist, cardiologist | V |
| ambig_12 | autoimmune hepatitis with thyroid disease | gastroenterologist | gastroenterologist, endocrinologist | V |
| ambig_13 | C. difficile colitis post-antibiotic | infectionist | infectionist, gastroenterologist | V |
| ambig_14 | liver cirrhosis with spontaneous bacterial peritonitis | infectionist | gastroenterologist, infectionist | V |
