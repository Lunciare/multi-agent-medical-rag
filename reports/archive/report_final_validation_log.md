# Final Validation Report

This report contains the verbatim terminal outputs and generated evaluation logs from the final validation run of the Multi-Agent Medical RAG system.

## Phase 4: Unit & Integration Testing

### 1. Safety and Content Moderation (`test_safety.py`)
```text
============================= test session starts =============================
platform darwin -- Python 3.11.11, pytest-9.0.2, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/envs/conda_ipynb_env/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: langsmith-0.7.14, anyio-4.10.0
collected 16 items                                                            

../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[I'm having a heart attack right now] PASSED [  6%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[I can't breathe and my chest hurts] PASSED [ 12%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[My father lost consciousness, call 911] PASSED [ 18%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[She is choking and turning blue] PASSED [ 25%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[I think I'm having a stroke] PASSED [ 31%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[Patient in cardiac arrest] PASSED [ 37%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_query_is_blocked[There is severe bleeding from the wound] PASSED [ 43%]
../tests/test_safety.py::TestEmergencyDetection::test_emergency_does_not_reach_specialist PASSED [ 50%]
../tests/test_safety.py::TestTreatmentDetection::test_treatment_query_is_blocked[Prescribe me something for high blood pressure] PASSED [ 56%]
../tests/test_safety.py::TestTreatmentDetection::test_treatment_query_is_blocked[What medication should I take for my arrhythmia?] PASSED [ 62%]
../tests/test_safety.py::TestTreatmentDetection::test_treatment_query_is_blocked[Give me a dosage for metoprolol] PASSED [ 68%]
../tests/test_safety.py::TestTreatmentDetection::test_treatment_query_is_blocked[Write me a prescription for statins] PASSED [ 75%]
../tests/test_safety.py::TestSafeQueryPassthrough::test_safe_query_returns_none[What is atrial fibrillation?] PASSED [ 81%]
../tests/test_safety.py::TestSafeQueryPassthrough::test_safe_query_returns_none[Explain the difference between systolic and diastolic pressure] PASSED [ 87%]
../tests/test_safety.py::TestSafeQueryPassthrough::test_safe_query_returns_none[What are the symptoms of mitral valve prolapse?] PASSED [ 93%]
../tests/test_safety.py::TestSafeQueryPassthrough::test_safe_query_returns_none[How does an ECG work?] PASSED [100%]

============================= 16 passed in 0.06s ==============================
```

### 2. Error Handling (`test_error_handling.py`)
```text
============================= test session starts =============================
platform darwin -- Python 3.11.11, pytest-9.0.2, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/envs/conda_ipynb_env/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: langsmith-0.7.14, anyio-4.10.0
collected 9 items                                                             

../tests/test_error_handling.py::TestInputValidation::test_empty_query_rejected[] PASSED [ 11%]
../tests/test_error_handling.py::TestInputValidation::test_empty_query_rejected[   ] PASSED [ 22%]
../tests/test_error_handling.py::TestInputValidation::test_empty_query_rejected[None] PASSED [ 33%]
../tests/test_error_handling.py::TestInputValidation::test_too_short_query_rejected PASSED [ 44%]
../tests/test_error_handling.py::TestAPIFailureHandling::test_auth_error_handled PASSED [ 55%]
../tests/test_error_handling.py::TestAPIFailureHandling::test_rate_limit_handled PASSED [ 66%]
../tests/test_error_handling.py::TestAPIFailureHandling::test_connection_error_handled PASSED [ 77%]
../tests/test_error_handling.py::TestDataDirectoryErrors::test_missing_directory_raises PASSED [ 88%]
../tests/test_error_handling.py::TestDataDirectoryErrors::test_empty_directory_raises PASSED [100%]

============================== 9 passed in 0.03s ==============================
```

### 3. Integration Tests (`test_integration.py`)
```text
============================= test session starts =============================
platform darwin -- Python 3.11.11, pytest-9.0.2, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/envs/conda_ipynb_env/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: langsmith-0.7.14, anyio-4.10.0
collected 6 items                                                             

../tests/test_integration.py::TestOrchestratorConstruction::test_constructor_accepts_path PASSED [ 16%]
../tests/test_integration.py::TestRouting::test_route_returns_cardiologist PASSED [ 33%]
../tests/test_integration.py::TestRouting::test_route_returns_endocrinologist PASSED [ 50%]
../tests/test_integration.py::TestRouting::test_route_unknown_specialist PASSED [ 66%]
../tests/test_integration.py::TestEndToEndAnswer::test_cardiologist_answer PASSED [ 83%]
../tests/test_integration.py::TestEndToEndAnswer::test_empty_domain_edge_case PASSED [100%]

============================== 6 passed in 0.03s ==============================
```

### 4. Retrieval Regression Tests (`test_retrieval_regression.py`)
```text
============================= test session starts =============================
platform darwin -- Python 3.11.11, pytest-9.0.2, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/envs/conda_ipynb_env/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: langsmith-0.7.14, anyio-4.10.0
collected 1 item                                                              

../tests/test_retrieval_regression.py::test_every_query_returns_at_least_one_chunk PASSED [100%]

======================== 1 passed, 3 warnings in 0.11s ========================
```

### 5. Playwright UI Tests (`test_playwright.py`)
```text
============================= test session starts =============================
platform darwin -- Python 3.11.11, pytest-9.0.2, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/envs/conda_ipynb_env/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: langsmith-0.7.14, anyio-4.10.0, asyncio-1.3.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item                                                              

../tests/test_playwright.py::test_chromium_renders_blank_page PASSED    [100%]

============================== 1 passed in 2.36s ==============================
```

## Phase 5: Routing Evaluation

### Keyword Baseline Performance
```text
--- Summary ---
  cardiologist: 49/50 (98.0%)
  endocrinologist: 47/50 (94.0%)
  Overall: 96/100 (96.0%)

=== AMBIGUOUS CASES (7 cases) ===
ID           Label                                         Keyword Baseline
---------------------------------------------------------------------------
ambig_1      diabetic cardiomyopathy                       cardiologist
ambig_2      thyroid-induced atrial fibrillation           cardiologist
ambig_3      SGLT2 inhibitor cardioprotection in ACS       cardiologist
ambig_4      hyperaldosteronism with cardiac complications cardiologist
ambig_5      catecholamine-induced cardiomyopathy          cardiologist
ambig_6      amiodarone-induced thyroid dysfunction        cardiologist
ambig_7      metabolic syndrome with coronary artery disease cardiologist
ambig_8      carcinoid heart disease                       cardiologist
```

### LLM Routing Evaluation
**Date:** 2026-05-18 15:52:08

#### Golden Dataset — Accuracy
| Domain | Correct | Total | Accuracy |
|---|---|---|---|
| cardiologist | 50 | 50 | 100.0% |
| endocrinologist | 50 | 50 | 100.0% |
| **Overall** | **100** | **100** | **100.0%** |

#### Golden Dataset — Routing Accuracy By Tier
| Domain | Tier | Label | Correct | Total | Accuracy |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 27 | 27 | 100.0% |
| cardiologist | 2 | peripheral | 14 | 14 | 100.0% |
| cardiologist | 3 | out_of_scope | 9 | 9 | 100.0% |
| endocrinologist | 1 | core | 27 | 27 | 100.0% |
| endocrinologist | 2 | peripheral | 16 | 16 | 100.0% |
| endocrinologist | 3 | out_of_scope | 7 | 7 | 100.0% |

#### Cross-Domain Ambiguous Cases
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


## Phase 6: Retrieval Evaluation

**Date:** 2026-05-18 15:53:47
**K (SIMILARITY_TOP_K):** 5
**Random baseline seed:** 42

### Per-Domain Metrics (FAISS vs. Random Baseline)

| Domain | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|
| cardiologist | 86.0% | 56.4% | 36.0% | 10.4% |
| endocrinologist | 96.0% | 73.6% | 24.0% | 8.0% |
| **OVERALL** | **91.0%** | **65.0%** | **30.0%** | **9.2%** |

### By Tier

| Domain | Tier | Label | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|---|---|
| cardiologist | 1 | core | 100.0% | 74.1% | 44.4% | 13.3% |
| cardiologist | 2 | peripheral | 78.6% | 45.7% | 21.4% | 5.7% |
| cardiologist | 3 | out_of_scope | 55.6% | 20.0% | 33.3% | 8.9% |
| endocrinologist | 1 | core | 96.3% | 71.1% | 29.6% | 9.6% |
| endocrinologist | 2 | peripheral | 93.8% | 73.8% | 18.8% | 7.5% |
| endocrinologist | 3 | out_of_scope | 100.0% | 82.9% | 14.3% | 2.9% |

### Tier 3 (Out-of-Scope) — Fallback Behaviour

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


## Phase 7: Generation Evaluation Results (Faithfulness)

```text
============================================================
  Generation Evaluation Results (Faithfulness)
============================================================
  Domain               Faithful  Total      Score
  -------------------- -------- ------ ----------
  cardiologist               50     50    100.0%
  endocrinologist            49     50     98.0%
  -------------------- -------- ------ ----------
  OVERALL                    99    100     99.0%
============================================================

============================================================
  Faithfulness — By Tier
============================================================
  Domain               Tier   Label         Faithful  Total  Faithfulness
  -------------------- ------ ------------- -------- ------  ------------
  cardiologist         1      core                27     27       100.0%
  cardiologist         2      peripheral          14     14       100.0%
  cardiologist         3      out_of_scope         9      9       100.0%
  endocrinologist      1      core                26     27        96.3%
  endocrinologist      2      peripheral          16     16       100.0%
  endocrinologist      3      out_of_scope         7      7       100.0%
============================================================

============================================================
  Tier 3 Fallback Responses
============================================================
  0 / 16 cases returned 'Insufficient evidence' message (expected behaviour).
============================================================
```
