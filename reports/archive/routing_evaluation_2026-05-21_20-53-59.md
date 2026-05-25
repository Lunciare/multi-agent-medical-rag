# Routing Evaluation Report

**Date:** 2026-05-21 20:53:59

## Golden Dataset — Accuracy (Wilson 95% CI)

| Domain | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| cardiologist | 18 | 18 | 100.0% [82.4%–100.0%] |
| endocrinologist | 22 | 23 | 95.7% [79.0%–99.2%] |
| gastroenterologist | 11 | 11 | 100.0% [74.1%–100.0%] |
| infectionist | 12 | 12 | 100.0% [75.8%–100.0%] |
| **Overall** | **63** | **64** | **98.4% [91.7%–99.7%]** |

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
| adv_amb_2 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_5 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_8 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_9 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_13 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_dom_15 | cardiologist | cardiologist | {"specialist": "cardiologist"} | V |
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
| adv_amb_6 | endocrinologist | gastroenterologist | {"specialist": "gastroenterologist"} | X |
| adv_amb_7 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_11 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_12 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_14 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_dom_16 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_13 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_amb_16 | endocrinologist | endocrinologist | {"specialist": "endocrinologist"} | V |
| adv_miss_9 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_miss_10 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_miss_11 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_miss_12 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_lang_9 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_lang_10 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_lang_11 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_lang_12 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_amb_10 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_amb_12 | gastroenterologist | cardiologist | {"specialist": "cardiologist"} | V |
| adv_amb_15 | gastroenterologist | gastroenterologist | {"specialist": "gastroenterologist"} | V |
| adv_miss_13 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_miss_14 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_miss_15 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_miss_16 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_lang_13 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_lang_14 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_lang_15 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_lang_16 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_dom_10 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_amb_9 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_amb_11 | infectionist | infectionist | {"specialist": "infectionist"} | V |
| adv_amb_14 | infectionist | infectionist | {"specialist": "infectionist"} | V |

## Adversarial Routing — Per-Category Accuracy

Categories: `misspelled` (typos that obscure standard terms), `non_english` (queries in Russian / French / Spanish), `dominant_pathology_mismatch` (surface vocabulary points one way but the actionable pathology is the other), `symptom_only_ambiguous` (symptom-only queries — `valid_domains` permits either specialty).

| Category | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| dominant_pathology_mismatch | 16 | 16 | 100.0% [80.6%–100.0%] |
| misspelled | 16 | 16 | 100.0% [80.6%–100.0%] |
| non_english | 16 | 16 | 100.0% [80.6%–100.0%] |
| symptom_only_ambiguous | 15 | 16 | 93.8% [71.7%–98.9%] |

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
| adv_dom_9 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_10 | dominant_pathology_mismatch | infectionist | infectionist | V |
| adv_dom_11 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_12 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_13 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_14 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_dom_15 | dominant_pathology_mismatch | cardiologist | cardiologist | V |
| adv_dom_16 | dominant_pathology_mismatch | endocrinologist | endocrinologist | V |
| adv_miss_1 | misspelled | cardiologist | cardiologist | V |
| adv_miss_2 | misspelled | cardiologist | cardiologist | V |
| adv_miss_3 | misspelled | cardiologist | cardiologist | V |
| adv_miss_4 | misspelled | cardiologist | cardiologist | V |
| adv_miss_5 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_6 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_7 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_8 | misspelled | endocrinologist | endocrinologist | V |
| adv_miss_9 | misspelled | gastroenterologist | gastroenterologist | V |
| adv_miss_10 | misspelled | gastroenterologist | gastroenterologist | V |
| adv_miss_11 | misspelled | gastroenterologist | gastroenterologist | V |
| adv_miss_12 | misspelled | gastroenterologist | gastroenterologist | V |
| adv_miss_13 | misspelled | infectionist | infectionist | V |
| adv_miss_14 | misspelled | infectionist | infectionist | V |
| adv_miss_15 | misspelled | infectionist | infectionist | V |
| adv_miss_16 | misspelled | infectionist | infectionist | V |
| adv_lang_1 | non_english | cardiologist | cardiologist | V |
| adv_lang_2 | non_english | cardiologist | cardiologist | V |
| adv_lang_3 | non_english | cardiologist | cardiologist | V |
| adv_lang_4 | non_english | cardiologist | cardiologist | V |
| adv_lang_5 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_6 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_7 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_8 | non_english | endocrinologist | endocrinologist | V |
| adv_lang_9 | non_english | gastroenterologist | gastroenterologist | V |
| adv_lang_10 | non_english | gastroenterologist | gastroenterologist | V |
| adv_lang_11 | non_english | gastroenterologist | gastroenterologist | V |
| adv_lang_12 | non_english | gastroenterologist | gastroenterologist | V |
| adv_lang_13 | non_english | infectionist | infectionist | V |
| adv_lang_14 | non_english | infectionist | infectionist | V |
| adv_lang_15 | non_english | infectionist | infectionist | V |
| adv_lang_16 | non_english | infectionist | infectionist | V |
| adv_amb_1 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_2 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_3 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_4 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_5 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_6 | symptom_only_ambiguous | cardiologist,endocrinologist | gastroenterologist | X |
| adv_amb_7 | symptom_only_ambiguous | cardiologist,endocrinologist | endocrinologist | V |
| adv_amb_8 | symptom_only_ambiguous | cardiologist,endocrinologist | cardiologist | V |
| adv_amb_9 | symptom_only_ambiguous | gastroenterologist,infectionist | infectionist | V |
| adv_amb_10 | symptom_only_ambiguous | gastroenterologist,infectionist | gastroenterologist | V |
| adv_amb_11 | symptom_only_ambiguous | cardiologist,infectionist | infectionist | V |
| adv_amb_12 | symptom_only_ambiguous | cardiologist,gastroenterologist | cardiologist | V |
| adv_amb_13 | symptom_only_ambiguous | endocrinologist,gastroenterologist | endocrinologist | V |
| adv_amb_14 | symptom_only_ambiguous | endocrinologist,infectionist | infectionist | V |
| adv_amb_15 | symptom_only_ambiguous | gastroenterologist,infectionist | gastroenterologist | V |
| adv_amb_16 | symptom_only_ambiguous | endocrinologist,gastroenterologist | endocrinologist | V |

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
