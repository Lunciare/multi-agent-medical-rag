# Stage: New Agents — Registry, Routing, and Documentation Updates

(Scope: bring the 4-specialist registry shipped in PR #1 — cardiologist,
endocrinologist, gastroenterologist, infectionist — to a tested,
documented, attributed state without building the two new FAISS
indices. Per-spec constraint: no numerical results in `report_final.md`
were modified; only prose descriptions of scope.)

## 1. What Was Tested

### Part 1 — Registry schema (no FAISS required)

**Inline check** (verbatim per spec):

```
$ cd multi-agent_system
$ python3 -c "
import sys, os
sys.path.insert(0, '.')
from agents.registry import AGENT_REGISTRY

REQUIRED = {'name', 'folder_path', 'role_prompt', 'domain_scope'}
EXPECTED_SPECIALISTS = {
    'cardiologist', 'endocrinologist', 'gastroenterologist', 'infectionist'
}

print(f'Specialists registered: {sorted(AGENT_REGISTRY.keys())}')
assert set(AGENT_REGISTRY.keys()) == EXPECTED_SPECIALISTS, \
    f'FAIL: expected {EXPECTED_SPECIALISTS}, got {set(AGENT_REGISTRY.keys())}'

for key, cfg in AGENT_REGISTRY.items():
    missing = REQUIRED - set(cfg.keys())
    extra   = set(cfg.keys()) - REQUIRED
    assert not missing, f'{key}: missing fields {missing}'
    assert not extra,   f'{key}: unexpected fields {extra}'
    assert cfg['name'].strip(),              f'{key}: name is empty'
    assert len(cfg['domain_scope']) >= 20,   f'{key}: domain_scope too short'
    assert len(cfg['role_prompt']) >= 500,   f'{key}: role_prompt too short'
    print(f'  [{key}] schema OK')

print()
print('ALL REGISTRY SCHEMA CHECKS PASSED')
"
Specialists registered: ['cardiologist', 'endocrinologist', 'gastroenterologist', 'infectionist']
  [cardiologist] schema OK
  [endocrinologist] schema OK
  [gastroenterologist] schema OK
  [infectionist] schema OK

ALL REGISTRY SCHEMA CHECKS PASSED
```

**PASS.**

**`tests/test_registry.py`** existed from the Stage 28 2-specialist version
and was overwritten with the spec's 4-specialist content (full content
in §3 below).

```
$ python -m pytest tests/test_registry.py -v --tb=short
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-9.0.2, pluggy-1.5.0 -- /opt/homebrew/Caskroom/miniconda/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: anyio-4.12.1, asyncio-1.3.0, langsmith-0.8.3
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 7 items

tests/test_registry.py::test_registry_has_all_four_specialists PASSED    [ 14%]
tests/test_registry.py::test_every_entry_has_required_fields_only PASSED [ 28%]
tests/test_registry.py::test_every_name_is_nonempty PASSED               [ 42%]
tests/test_registry.py::test_every_domain_scope_is_substantial PASSED    [ 57%]
tests/test_registry.py::test_every_role_prompt_is_substantial PASSED     [ 71%]
tests/test_registry.py::test_new_specialists_have_correct_keys PASSED    [ 85%]
tests/test_registry.py::test_new_role_prompts_contain_critical_rule PASSED [100%]

============================== 7 passed in 0.01s ===============================
```

**PASS.** 7 tests, all green. Note: the Stage 28 file had 6 tests; the
4-specialist replacement adds `test_new_specialists_have_correct_keys`
and `test_new_role_prompts_contain_critical_rule` (the new specialists'
role prompts both contain the `CRITICAL_RULE` block and the
"Insufficient evidence" fallback phrase — both confirmed) while dropping
`test_every_folder_path_exists` (the spec's replacement schema doesn't
include that test).

### Part 2 — Routing prompt coverage + ambiguous cases

```
$ cd multi-agent_system
$ python3 -c "
import sys
sys.path.insert(0, '.')
from agents.registry import AGENT_REGISTRY

keys = sorted(AGENT_REGISTRY.keys())
routing_str = ' or '.join(keys)
print(f'Routing prompt specialist list: {routing_str}')
assert 'gastroenterologist' in routing_str, 'FAIL: gastroenterologist missing'
assert 'infectionist'       in routing_str, 'FAIL: infectionist missing'
assert 'cardiologist'       in routing_str, 'FAIL: cardiologist missing'
assert 'endocrinologist'    in routing_str, 'FAIL: endocrinologist missing'
print('Routing prompt coverage: PASS')
"
Routing prompt specialist list: cardiologist or endocrinologist or gastroenterologist or infectionist
Routing prompt coverage: PASS
```

**PASS.** The orchestrator's `_routing_system_prompt` (Stage 24
implementation) iterates `self.allowed_specialists` and inlines each
agent's `domain_scope`, so it auto-expands from 2 → 4 specialists with
zero code changes.

**Ambiguous-cases JSON validation** (after appending the 6 new cases —
schema deviation note: the spec gave new cases with `valid_domains`
field while existing cases use `domains`; I included both fields with
identical values in each new case so the spec's verbatim shape is
preserved AND the existing `evaluate_routing.py` code path that reads
`case["domains"]` keeps working):

```
$ python3 -c "
import json
data = json.load(open('multi-agent_system/tests/data/ambiguous_cases.json'))
print(f'Total ambiguous cases: {len(data)}')
ids = [c['id'] for c in data]
print(f'IDs: {ids}')
assert len(ids) == len(set(ids)), 'DUPLICATE IDs FOUND'
new_ids = {'ambig_9','ambig_10','ambig_11','ambig_12','ambig_13','ambig_14'}
assert new_ids.issubset(set(ids)), f'Missing new cases: {new_ids - set(ids)}'
print('ambiguous_cases.json: VALID')
"
Total ambiguous cases: 14
IDs: ['ambig_1', 'ambig_2', 'ambig_3', 'ambig_4', 'ambig_5', 'ambig_6', 'ambig_7', 'ambig_8', 'ambig_9', 'ambig_10', 'ambig_11', 'ambig_12', 'ambig_13', 'ambig_14']
ambiguous_cases.json: VALID
```

**PASS.** 8 existing + 6 new = 14 unique IDs.

### Part 3 — Settings/path consistency

```
$ cd multi-agent_system
$ python3 -c "
import sys, os
sys.path.insert(0, '.')
from settings import (
    GASTRO_KNOWLEDGE_BASE_DIR,
    INFECTIONIST_KNOWLEDGE_BASE_DIR,
)
from agents.registry import AGENT_REGISTRY

print(f'GASTRO_KNOWLEDGE_BASE_DIR   = {GASTRO_KNOWLEDGE_BASE_DIR}')
print(f'INFECTIONIST_KNOWLEDGE_BASE_DIR = {INFECTIONIST_KNOWLEDGE_BASE_DIR}')

gastro_cfg = AGENT_REGISTRY['gastroenterologist']
infect_cfg = AGENT_REGISTRY['infectionist']

assert gastro_cfg['folder_path'] == GASTRO_KNOWLEDGE_BASE_DIR, \
    f'PATH MISMATCH: registry={gastro_cfg[\"folder_path\"]} settings={GASTRO_KNOWLEDGE_BASE_DIR}'
assert infect_cfg['folder_path'] == INFECTIONIST_KNOWLEDGE_BASE_DIR, \
    f'PATH MISMATCH: registry={infect_cfg[\"folder_path\"]} settings={INFECTIONIST_KNOWLEDGE_BASE_DIR}'

print('Path consistency: PASS')

for key, path in [
    ('gastroenterologist', GASTRO_KNOWLEDGE_BASE_DIR),
    ('infectionist',       INFECTIONIST_KNOWLEDGE_BASE_DIR),
]:
    exists = os.path.isdir(path)
    print(f'  {key} data dir: {\"EXISTS\" if exists else \"MISSING (FAISS not built yet — expected)\"}')
"
GASTRO_KNOWLEDGE_BASE_DIR   = /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag/data/processed/gastroenterologist
INFECTIONIST_KNOWLEDGE_BASE_DIR = /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag/data/processed/infection
Path consistency: PASS
  gastroenterologist data dir: EXISTS
  infectionist data dir: EXISTS
```

**PASS.** Both data directories exist on disk (the corpora are
committed); only the FAISS *indices* are pending. The path constants
match the registry's `folder_path` exactly — confirming that the prior
commit `a8667f27f Fix gastroenterologist and infectionist folder_path
mismatch + script-filename typos` resolved the original PR's path bug
correctly.

### Part 4 — Full pytest suite

```
$ python -m pytest tests/ -v --tb=short
... (per-test PASSED lines elided for brevity)
================== 41 passed, 1 skipped, 8 warnings in 25.60s ==================
```

**PASS — no regressions.** Pass count is 41 (up from the Stage 33
baseline of 40) because the new `tests/test_registry.py` has 7 tests
vs the prior 6. The 1 skipped is the pre-existing `test_crawler_imports.py`
Playwright skip (browser-binary precondition unrelated to this PR).
The 8 warnings are upstream-dep deprecations (Gradio + Pandas), all
pre-existing.

### Per-suite summary

| Test file | Pre-PR baseline | Post-Stage | Δ |
|---|---|---|---|
| `test_crawler_imports.py` | 0+1 skipped | 0+1 skipped | unchanged |
| `test_error_handling.py` | 7 passed | 7 passed | unchanged |
| `test_integration.py` | 6 passed | 6 passed | unchanged |
| `test_playwright.py` | 1 passed | 1 passed | unchanged |
| `test_registry.py` | 6 passed | **7 passed** | **+1** (new test added) |
| `test_retrieval_regression.py` | 2 passed | 2 passed | unchanged |
| `test_safety.py` | 18 passed | 18 passed | unchanged |
| **Total** | **40 passed, 1 skipped** | **41 passed, 1 skipped** | **+1** |

## 2. Files Updated

| File | Change |
|---|---|
| `tests/test_registry.py` | Replaced 6-test (2-specialist) Stage 28 content with 7-test (4-specialist) spec content. Adds explicit checks that gastroenterologist + infectionist are present, both `role_prompt`s contain the `CRITICAL_RULE` block, and the "Insufficient evidence" fallback phrase. Drops `test_every_folder_path_exists` (not in the spec's replacement). |
| `multi-agent_system/tests/data/ambiguous_cases.json` | Appended 6 new cross-domain cases (`ambig_9..ambig_14`) for the new specialist combinations: GI×endo, ID×endo, ID×cardio, GI×endo (autoimmune-hepatitis variant), ID×GI, GI×ID. Each new case carries both `valid_domains` (spec's field name) and `domains` (existing `evaluate_routing.py` field name) with identical values, so the spec's literal shape is preserved without breaking the existing ambiguous-cases code path. |
| `README.md` (4 edits) | (a) "Two specialists are implemented:" → "Four specialists are registered ..." with gastro/infect bullets marked `index pending`. (b) Architecture ASCII diagram routing line now reads `cardiology / endocrinology / gastroenterology / infectiology`. (c) Repository Structure tree under `data/processed/` adds `gastroenterologist/` and `infection/` rows annotated `data committed; FAISS index pending`. (d) Limitations bullet "Two-agent scope" replaced with "Four specialists registered; two FAISS indices pending" — same link target to report §6 L3. |
| `reports/report_final.md` (2 edits) | (a) §6 Limitation 3 rewritten: now states four specialists are registered, two FAISS indices pending, with the exact build commands. (b) §7 Conclusion's final paragraph: "Adding a third specialty" → explicit four-specialist statement plus a §6 L3 back-reference. **No numerical result in §4 tables was changed.** |
| `multi-agent_system/tests/evaluate_routing_baseline.py` | Added a 6-line comment above `CARDIO_KEYWORDS = {...}` noting the keyword baseline is two-specialty-only (cardiology + endocrinology) and that the §4.1 baseline accuracy in `report_final.md` is for the original two-agent system. |
| `README.md` (Contributors section) | New section inserted between the disclaimer paragraph and `## Overview`, with the verbatim 2-row table per spec. |
| `reports/report_stage_new_agents.md` | This stage report (new). |

## 3. New Test File

Full content of `tests/test_registry.py`:

```python
"""Schema-validation tests for agents.registry.AGENT_REGISTRY.
Validates all four specialists. Does not require FAISS indices.
"""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "multi-agent_system"))
import pytest

@pytest.fixture(scope="module")
def registry():
    from agents.registry import AGENT_REGISTRY
    return AGENT_REGISTRY

REQUIRED_FIELDS = {"name", "folder_path", "role_prompt", "domain_scope"}
EXPECTED_KEYS   = {"cardiologist", "endocrinologist",
                   "gastroenterologist", "infectionist"}

def test_registry_has_all_four_specialists(registry):
    assert set(registry.keys()) == EXPECTED_KEYS, \
        f"Expected {EXPECTED_KEYS}, got {set(registry.keys())}"

def test_every_entry_has_required_fields_only(registry):
    for key, cfg in registry.items():
        missing = REQUIRED_FIELDS - set(cfg.keys())
        extra   = set(cfg.keys()) - REQUIRED_FIELDS
        assert not missing, f"{key}: missing {missing}"
        assert not extra,   f"{key}: unexpected {extra}"

def test_every_name_is_nonempty(registry):
    for key, cfg in registry.items():
        assert isinstance(cfg["name"], str) and cfg["name"].strip(), \
            f"{key}: name empty"

def test_every_domain_scope_is_substantial(registry):
    for key, cfg in registry.items():
        assert len(cfg["domain_scope"].strip()) >= 20, \
            f"{key}: domain_scope too short"

def test_every_role_prompt_is_substantial(registry):
    for key, cfg in registry.items():
        assert len(cfg["role_prompt"]) >= 500, \
            f"{key}: role_prompt too short ({len(cfg['role_prompt'])} chars)"

def test_new_specialists_have_correct_keys(registry):
    assert "gastroenterologist" in registry
    assert "infectionist" in registry

def test_new_role_prompts_contain_critical_rule(registry):
    for key in ("gastroenterologist", "infectionist"):
        assert "CRITICAL_RULE" in registry[key]["role_prompt"], \
            f"{key}: CRITICAL_RULE block missing from role_prompt"
        assert "Insufficient evidence" in registry[key]["role_prompt"], \
            f"{key}: fallback phrase missing"
```

## 4. New Ambiguous Cases

Full updated `multi-agent_system/tests/data/ambiguous_cases.json` (14 total: 8 unchanged from prior + 6 new at the tail):

```json
[
    {
        "id": "ambig_1",
        "query": "A 58-year-old male with poorly controlled type 2 diabetes and HbA1c of 9.2% presents with progressive exertional dyspnea and bilateral ankle edema. Echocardiogram shows an ejection fraction of 40% with diastolic dysfunction. What is the relationship between his diabetes and cardiac findings?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "diabetic cardiomyopathy",
        "notes": "Diabetic cardiomyopathy is managed jointly by cardiology (heart failure) and endocrinology (glycemic control)."
    },
    {
        "id": "ambig_2",
        "query": "A 45-year-old female with Graves' disease presents with new-onset atrial fibrillation, resting heart rate of 130 bpm, and palpitations. TSH is undetectable and free T4 is elevated. How should both the arrhythmia and the underlying thyroid disorder be managed?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "thyroid-induced atrial fibrillation",
        "notes": "Hyperthyroidism is the root cause (endocrinology) but the acute arrhythmia requires cardiac management."
    },
    {
        "id": "ambig_3",
        "query": "A 62-year-old male with a 15-year history of type 2 diabetes on metformin and empagliflozin presents with chest pain. ECG shows ST depression in leads V4-V6. His last HbA1c was 7.1%. What is the interplay between his diabetic medications and his acute coronary event?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "SGLT2 inhibitor cardioprotection in ACS",
        "notes": "SGLT2 inhibitors have proven cardiovascular benefits; the case requires both cardiac workup and diabetes medication review."
    },
    {
        "id": "ambig_4",
        "query": "A 50-year-old female with primary hyperaldosteronism (Conn's syndrome) presents with therapy-resistant hypertension despite three antihypertensive agents, hypokalemia (K+ 2.9 mEq/L), and ECG showing prominent U waves. How should the endocrine and cardiovascular aspects be addressed?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "hyperaldosteronism with cardiac complications",
        "notes": "Primary aldosteronism is endocrine in origin but causes hypertension, hypokalemia, and arrhythmia risk requiring cardiovascular management."
    },
    {
        "id": "ambig_5",
        "query": "A 35-year-old female with known pheochromocytoma presents to the ED with acute hypertensive crisis (BP 240/140 mmHg), severe headache, diaphoresis, and ECG showing ST changes consistent with stress cardiomyopathy (Takotsubo). What is the management priority?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "catecholamine-induced cardiomyopathy",
        "notes": "Pheochromocytoma (endocrine tumor) causes catecholamine surge leading to cardiac damage."
    },
    {
        "id": "ambig_6",
        "query": "A 70-year-old male on long-term amiodarone for recurrent ventricular tachycardia develops fatigue, weight gain, and a TSH of 45 mIU/L. Free T4 is low. What are the considerations for managing amiodarone-induced hypothyroidism without compromising arrhythmia control?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "amiodarone-induced thyroid dysfunction",
        "notes": "Amiodarone is essential for arrhythmia control but causes thyroid dysfunction; both specialists must coordinate."
    },
    {
        "id": "ambig_7",
        "query": "A 55-year-old obese female with metabolic syndrome (BMI 38, fasting glucose 130 mg/dL, triglycerides 280 mg/dL) presents with exertional angina and a positive stress test. She is being considered for bariatric surgery. How do the metabolic and cardiovascular risks interact?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "metabolic syndrome with coronary artery disease",
        "notes": "Metabolic syndrome drives both insulin resistance and atherosclerosis; treatment requires coordinated metabolic and cardiac care."
    },
    {
        "id": "ambig_8",
        "query": "A 25-year-old male presents with episodic flushing, diarrhea, and wheezing. 24-hour urine 5-HIAA is 68 mg/day (normal < 15). CT abdomen shows a 2 cm ileal mass with liver lesions. What syndrome has developed and what systemic complication requires cardiac evaluation?",
        "domains": ["cardiologist", "endocrinologist"],
        "label": "carcinoid heart disease",
        "notes": "Carcinoid syndrome is an endocrine tumor issue, but carcinoid heart disease (valvular lesions) requires cardiac evaluation."
    },
    {
        "id": "ambig_9",
        "label": "H. pylori peptic ulcer with iron deficiency",
        "query": "A 45-year-old male with confirmed H. pylori peptic ulcer presents with iron deficiency anaemia and fatigue. Which specialist should manage the combined workup?",
        "valid_domains": ["gastroenterologist", "endocrinologist"],
        "domains": ["gastroenterologist", "endocrinologist"]
    },
    {
        "id": "ambig_10",
        "label": "sepsis with new-onset hyperglycaemia",
        "query": "A patient in the ICU with gram-negative sepsis develops new hyperglycaemia with glucose 18 mmol/L and no prior diabetes history. Which specialist should guide management?",
        "valid_domains": ["infectionist", "endocrinologist"],
        "domains": ["infectionist", "endocrinologist"]
    },
    {
        "id": "ambig_11",
        "label": "HIV with cardiac complications",
        "query": "A 38-year-old HIV-positive patient on antiretroviral therapy presents with dyspnea and echocardiographic findings of dilated cardiomyopathy. Routing decision?",
        "valid_domains": ["infectionist", "cardiologist"],
        "domains": ["infectionist", "cardiologist"]
    },
    {
        "id": "ambig_12",
        "label": "autoimmune hepatitis with thyroid disease",
        "query": "A 34-year-old female with autoimmune hepatitis and elevated TSH is referred for joint management. Which specialist takes the lead?",
        "valid_domains": ["gastroenterologist", "endocrinologist"],
        "domains": ["gastroenterologist", "endocrinologist"]
    },
    {
        "id": "ambig_13",
        "label": "C. difficile colitis post-antibiotic",
        "query": "A patient develops severe watery diarrhoea and abdominal cramps after a 10-day course of clindamycin. Stool toxin assay is positive. Which specialist manages this?",
        "valid_domains": ["infectionist", "gastroenterologist"],
        "domains": ["infectionist", "gastroenterologist"]
    },
    {
        "id": "ambig_14",
        "label": "liver cirrhosis with spontaneous bacterial peritonitis",
        "query": "A 58-year-old male with decompensated liver cirrhosis develops fever, abdominal pain and ascitic fluid polymorphs of 350 cells/mm3. Which specialist should lead?",
        "valid_domains": ["gastroenterologist", "infectionist"],
        "domains": ["gastroenterologist", "infectionist"]
    }
]
```

## 5. Known Gaps (FAISS indices not yet built)

- **`gastroenterologist` FAISS index: NOT BUILT** — data present at
  `data/processed/gastroenterologist/`, build with:
  ```
  python multi-agent_system/build_index.py --specialty gastroenterologist
  ```
- **`infectionist` FAISS index: NOT BUILT** — data present at
  `data/processed/infection/` (note: directory is `infection/`, not
  `infectionist/` — the registry's `INFECTIONIST_KNOWLEDGE_BASE_DIR`
  in `settings.py` points to the correct `infection/` path), build with:
  ```
  python multi-agent_system/build_index.py --specialty infectionist
  ```
- **`evaluate_retrieval.py`, `evaluate_generation.py`,
  `evaluate_chunk_relevance.py` for the new agents: BLOCKED** until
  indices are built. These scripts still iterate
  `("cardiologist", "endocrinologist")` tuples for their per-domain
  aggregation; extending them to the four-specialty registry is
  trivial (replace the literal tuple with
  `tuple(AGENT_REGISTRY.keys())`) but was intentionally left out of
  this stage per spec ("Do NOT modify any evaluation result tables or
  numerical results in report_final.md").
- **`evaluate_routing.py` for new agents: can run** as soon as
  `golden_dataset.json` is extended with gastroenterology and
  infectiology cases. The orchestrator's routing prompt (Stage 24
  implementation) already auto-expands to all four specialties, so
  the only blocker is dataset coverage. Two minor downstream code
  edits will be needed when this is done:
  (a) update `evaluate_routing.py:ROUTING_SYSTEM_PROMPT` (the
  evaluator's hardcoded prompt string is currently 2-specialist) and
  `ALLOWED_SPECIALISTS` (currently `{"cardiologist",
  "endocrinologist"}`); and
  (b) loop bodies that iterate `("cardiologist", "endocrinologist")`
  literally would benefit from the same `tuple(AGENT_REGISTRY.keys())`
  treatment.
- **`evaluate_routing_baseline.py` keyword dictionary**: marked
  inline with a NOTE comment that the baseline is two-specialty-only;
  extending requires adding `GASTRO_KEYWORDS` / `INFECT_KEYWORDS` sets
  and a routing rule with explicit precedence between them.
- **Existing pytest suite**: every test passes. The four-specialty
  registry didn't break any of the 40 pre-existing tests because none
  of them attempt to construct `SpecialistAgent` instances for the
  new specialists or load their FAISS indices.

