# Stage 28 — Registry Schema-Validation Tests

(Filename note: next sequential number — Stages 23 through 27 are already
taken: dict-access migration, `domain_scope` routing prompt, adversarial
routing test set, README refresh, MRR bootstrap CI.)

## 1. What Was Changed

Adds `tests/test_registry.py` — 6 schema-validation tests that fail fast
on misconfigured `AGENT_REGISTRY` entries. Catches at *test* time the
failure modes that previously surfaced only at runtime as opaque routing
or FAISS-load errors:

- a typo in a required field name (`role_promp` instead of `role_prompt`)
  would currently raise a `KeyError` deep inside `SpecialistAgent.__init__`
  on import — now caught by `test_every_entry_has_required_fields_only`;
- an `extra` field would silently slip into `SpecialistAgent(**cfg)` and
  fail with `TypeError: unexpected keyword argument` — now caught by the
  same test;
- a stale `folder_path` (e.g. specialty renamed, directory moved) would
  raise `FileNotFoundError` from `FAISS.load_local()` ~30 seconds into
  orchestrator construction — now caught by
  `test_every_folder_path_exists` in milliseconds;
- an empty `domain_scope` (or whitespace-only, or a single placeholder
  word) would silently degrade the Stage 24 routing prompt — now caught
  by `test_every_domain_scope_meets_minimum_length`;
- an accidentally-shortened `role_prompt` (e.g. forgetting to append
  `_RULES_AND_FORMAT`) would silently produce ungrounded answers — now
  caught by `test_every_role_prompt_is_substantial`.

## 2. Test File (verbatim, full content of `tests/test_registry.py`)

```python
"""Schema-validation tests for agents.registry.AGENT_REGISTRY.

A misconfigured registry entry would otherwise surface at runtime as an
indistinguishable routing failure. These tests fail-fast on:
  - missing or extra fields per entry,
  - non-existent folder_path,
  - empty name or empty / too-short domain_scope.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "multi-agent_system"))

import pytest


@pytest.fixture(scope="module")
def registry():
    from agents.registry import AGENT_REGISTRY
    return AGENT_REGISTRY


REQUIRED_FIELDS = {"name", "folder_path", "role_prompt", "domain_scope"}


def test_registry_has_at_least_two_specialists(registry):
    assert len(registry) >= 2, (
        f"AGENT_REGISTRY must contain ≥2 specialists, got {len(registry)}"
    )


def test_every_entry_has_required_fields_only(registry):
    for key, cfg in registry.items():
        missing = REQUIRED_FIELDS - set(cfg.keys())
        extra = set(cfg.keys()) - REQUIRED_FIELDS
        assert not missing, f"{key}: missing required fields {missing}"
        assert not extra,   f"{key}: unexpected extra fields {extra}"


def test_every_folder_path_exists(registry):
    for key, cfg in registry.items():
        assert os.path.isdir(cfg["folder_path"]), (
            f"{key}: folder_path {cfg['folder_path']!r} does not exist"
        )


def test_every_name_is_nonempty_string(registry):
    for key, cfg in registry.items():
        assert isinstance(cfg["name"], str) and cfg["name"].strip(), (
            f"{key}: name must be a non-empty string, got {cfg['name']!r}"
        )


def test_every_domain_scope_meets_minimum_length(registry):
    for key, cfg in registry.items():
        scope = cfg["domain_scope"]
        assert isinstance(scope, str) and len(scope.strip()) >= 10, (
            f"{key}: domain_scope must be ≥10 chars of meaningful text, "
            f"got {scope!r}"
        )


def test_every_role_prompt_is_substantial(registry):
    for key, cfg in registry.items():
        prompt = cfg["role_prompt"]
        assert isinstance(prompt, str) and len(prompt) >= 500, (
            f"{key}: role_prompt looks too short ({len(prompt)} chars); "
            f"the canonical prompt with _RULES_AND_FORMAT is ≥1500 chars"
        )
```

(The file matches the spec character-for-character — no liberties taken
on either the docstring, the fixture scope, or the threshold values.)

## 3. Pytest Output

### 3.1. Smoke test — file only (verbatim per spec)

Command:

```
python -m pytest tests/test_registry.py -v
```

Stdout:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-9.0.2, pluggy-1.5.0 -- /opt/homebrew/Caskroom/miniconda/base/bin/python
cachedir: .pytest_cache
rootdir: /Users/aleksandrasuvorova/Documents/GitHub/multi-agent-medical-rag
plugins: anyio-4.12.1, asyncio-1.3.0, langsmith-0.8.3
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 6 items

tests/test_registry.py::test_registry_has_at_least_two_specialists PASSED [ 16%]
tests/test_registry.py::test_every_entry_has_required_fields_only PASSED [ 33%]
tests/test_registry.py::test_every_folder_path_exists PASSED             [ 50%]
tests/test_registry.py::test_every_name_is_nonempty_string PASSED        [ 66%]
tests/test_registry.py::test_every_domain_scope_meets_minimum_length PASSED [ 83%]
tests/test_registry.py::test_every_role_prompt_is_substantial PASSED     [100%]

============================== 6 passed in 0.01s ===============================
```

**6 passed** — matches the spec's expected `6 passed`. Each assertion
fires against the current 2-specialist registry (cardiologist +
endocrinologist), and all four schema axes are exercised at least once
per entry.

### 3.2. Full suite (verbatim per spec)

Command:

```
python -m pytest tests/ -q
```

Tail of stdout:

```
... (3 SwigPyPacked DeprecationWarnings from a transitive dep — pre-existing)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
40 passed, 1 skipped, 3 warnings in 1.27s
```

## 4. New Total Test Count

| Metric | Pre-Stage-28 | Post-Stage-28 | Delta |
|---|---|---|---|
| Passed | 34 | **40** | **+6** |
| Skipped | 1 | 1 | 0 |
| Warnings | 3 | 3 | 0 |

The 6 new assertions are exactly the +6 the spec predicted; no
pre-existing test was disturbed (no flake in `test_safety.py`,
`test_error_handling.py`, `test_integration.py`,
`test_retrieval_regression.py`, `test_crawler_imports.py`, or
`test_playwright.py`). The 1 skipped test is the pre-existing
Playwright browser test that requires `playwright install chromium`.

## 5. Runtime Failure Mode Now Caught at Test Time

**One-line statement (per spec):** an `AGENT_REGISTRY` entry with a
mistyped field name, a stale or moved `folder_path`, an empty / single-word
`domain_scope`, or a truncated `role_prompt` will now fail
`tests/test_registry.py` in <100 ms at CI time instead of surfacing
later as a `KeyError` / `TypeError` / `FileNotFoundError` deep inside
`SpecialistAgent.__init__()` or as a silently-degraded routing prompt
(Stage 24) or silently-ungrounded answer (cf. the §3.5 keyword-stripping
caveat).

## 6. Files Touched

- `tests/test_registry.py` — **new** (74 lines, 6 tests)
- `reports/report_stage_28.md` — this stage report (new)
