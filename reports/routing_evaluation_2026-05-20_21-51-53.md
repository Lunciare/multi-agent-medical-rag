# Routing Evaluation Report

**Date:** 2026-05-20 21:51:53

## Golden Dataset — Accuracy (Wilson 95% CI)

| Domain | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| cardiologist | 15 | 15 | 100.0% [79.6%–100.0%] |
| endocrinologist | 17 | 17 | 100.0% [81.6%–100.0%] |
| **Overall** | **32** | **32** | **100.0% [89.3%–100.0%]** |

## Golden Dataset — Routing Accuracy By Tier (Wilson 95% CI)

| Domain | Tier | Label | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|---|---|

## Golden Dataset — Per-Query Details

| ID | Expected | Predicted | Raw LLM Output | Result |
|---|---|---|---|---|
| adv_miss_1 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_miss_2 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_miss_3 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_miss_4 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_lang_1 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_lang_2 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_lang_3 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_lang_4 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_1 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_5 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_7 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_1 | cardiologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_2 | cardiologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_5 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_8 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_miss_5 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_miss_6 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_miss_7 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_miss_8 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_lang_5 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_lang_6 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_lang_7 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_lang_8 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_2 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_3 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_4 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_6 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_8 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_3 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_4 | endocrinologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_6 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_7 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |

## Adversarial Routing — Per-Category Accuracy

Categories: `misspelled` (typos that obscure standard terms), `non_english` (queries in Russian / French / Spanish), `dominant_pathology_mismatch` (surface vocabulary points one way but the actionable pathology is the other), `symptom_only_ambiguous` (symptom-only queries — `valid_domains` permits either specialty).

| Category | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| dominant_pathology_mismatch | 8 | 8 | 100.0% [67.6%–100.0%] |
| misspelled | 8 | 8 | 100.0% [67.6%–100.0%] |
| non_english | 8 | 8 | 100.0% [67.6%–100.0%] |
| symptom_only_ambiguous | 8 | 8 | 100.0% [67.6%–100.0%] |

### Adversarial Routing — Per-Case Details

| ID | Category | Expected | Predicted | Correct? |
|---|---|---|---|---|
| adv_dom_1 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_2 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_3 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_4 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_5 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_6 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_7 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_8 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_miss_1 | misspelled | cardiologist | cardiologist | V |
| adv_miss_2 | misspelled | cardiologist | cardiologist | V |
| adv_miss_3 | misspelled | cardiologist | cardiologist | V |
| adv_miss_4 | misspelled | cardiologist | cardiologist | V |
| adv_miss_5 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_6 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_7 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_8 | misspelled | endocrinologist | endocrinologist | V |
| adv_lang_1 | non_english | cardiologist | cardiologist | V |
| adv_lang_2 | non_english | cardiologist | cardiologist | V |
| adv_lang_3 | non_english | cardiologist | cardiologist | V |
| adv_lang_4 | non_english | cardiologist | cardiologist | V |
| adv_lang_5 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_6 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_7 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_8 | non_english | endocrinologist | endocrinologist | V |
| adv_amb_1 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_2 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_3 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_4 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_5 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_6 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_7 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_8 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |

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
