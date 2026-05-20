# Stage 23 — Remove `MedicalOrchestrator` @property Shims; Migrate All Call Sites to `self.agents[key]`

## 1. What Was Changed

Stage 8 (§6 "Open Questions / Trade-offs") explicitly deferred this
migration: the registry/dispatch refactor switched `MedicalOrchestrator`
internals to a single `self.agents` dict but kept two `@property`
forwarders (`cardiologist`, `endocrinologist`) so the ~20 external call
sites in `tests/`, `evaluate_*.py`, `tune_*.py`,
`inspect_judge_disagreements.py`, and `annotate_gold_sources.py` would
continue to work unchanged. Stage 23 finishes that work.

Three categories of change:

1. **Call-site migration** — every external read of
   `orchestrator.cardiologist` / `orchestrator.endocrinologist` was
   rewritten to `orchestrator.agents["cardiologist"]` /
   `orchestrator.agents["endocrinologist"]`. Substring replacement was
   safe because no longer-named identifiers (e.g. `…cardiologists`) exist
   in the tree.
2. **`@property` removal** — the two `@property` shims in
   `multi-agent_system/orchestrator.py` and the four-line comment block
   above them were deleted. New code can no longer read these attributes;
   `KeyError` will surface immediately if a future caller forgets to use
   `agents[key]`.
3. **Diagnostic dead code removal** — `multi-agent_system/scratch.py` (a
   30-line one-off diagnostic that printed Tier 2 cardiology misses) was
   deleted. Confirmed unreferenced (no `import scratch` or
   `from scratch` anywhere in the tree).

## 2. Files Modified — Substitution Count Per File

| File | `orchestrator.cardiologist` subs | `orchestrator.endocrinologist` subs | Total | Notes |
|---|---:|---:|---:|---|
| `multi-agent_system/orchestrator.py` | — | — | — | Deleted the two `@property` defs + the 4-line "backward-compat aliases" comment block above them. |
| `multi-agent_system/tests/evaluate_chunk_relevance.py` | 1 | 1 | 2 | Lines 86, 88 (`if/elif` dispatch). |
| `multi-agent_system/tests/tune_refusal_gate.py` | 2 | 2 | 4 | Lines 70/72 (`if/elif` dispatch) and 138/144 (per-specialty corpus-stats load). |
| `multi-agent_system/tests/annotate_gold_sources.py` | 1 | 1 | 2 | Line 52/53 ternary. |
| `multi-agent_system/tests/evaluate_retrieval.py` | 4 | 4 | 8 | Lines 246/247 ternary; 382/383 + 384/385 `domain_pool` dict; 403/405 `if/elif` dispatch. Also patched the `domain_pool` truthiness check from `orchestrator.agents["x"]` to `orchestrator.agents.get("x")` — naive substitution would have raised `KeyError` in the ablation path where `_AblationOrchestrator.agents` carries only `{"cardiologist": …}`. |
| `multi-agent_system/tests/inspect_judge_disagreements.py` | 1 | 1 | 2 | Lines 104/105 ternary. |
| `multi-agent_system/tests/tune_retrieval.py` | 1 | 1 | 2 | Lines 50/51 ternary. |
| `multi-agent_system/tests/evaluate_generation.py` | 2 | 2 | 4 | Lines 138/140 (multi-judge `if/elif`) and 435/437 (legacy `if/elif`). |
| `multi-agent_system/tests/tune_chunk_size.py` | 0 | 0 | 0 | Listed in the spec but contained no `orchestrator.cardiologist/endocrinologist` patterns — only `"cardiologist"` / `"endocrinologist"` *string* literals as dataset-filter keys, which were left untouched. |
| `multi-agent_system/scratch.py` | — | — | — | **Deleted entirely** (30 lines). |

**Substitution total:** 24 across 7 test/eval files, plus 1 defensive
`.get()` patch in `evaluate_retrieval.py` to preserve the ablation path's
`if … else []` fallback behaviour.

## 3. Grep Smoke-Test Output

Command (verbatim from the spec):

```
grep -rn "orchestrator\.cardiologist\|orchestrator\.endocrinologist" \
     multi-agent_system/ tests/
```

Output:

```
(no matches)
EXIT: 1
```

`grep` exit code 1 = no match = success signal, as the spec calls out.

## 4. Pytest Summary Line

Command (verbatim from the spec):

```
python -m pytest tests/ -q          # run from repo root (the test suite
                                    # lives at `<repo>/tests/`, not at
                                    # `<repo>/multi-agent_system/tests/`,
                                    # which holds the evaluation scripts)
```

Output:

```
s..................................                                      [100%]
=============================== warnings summary ===============================
... (3 SwigPyPacked DeprecationWarnings from a transitive dep — pre-existing)
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
34 passed, 1 skipped, 3 warnings in 1.18s
```

**34 passed, 1 skipped** — exactly the spec's expected `34 passed`. The
single skip is pre-existing (a Playwright test that requires the browser
runtime) and is unrelated to this refactor. `scratch.py` was unreferenced
diagnostic code outside `tests/`, so its removal did not affect the test
count.

## 5. Direct Confirmation: Dev Routing Accuracy Unchanged

Command:

```
cd multi-agent_system
python tests/evaluate_routing.py --split dev > /tmp/routing_post_refactor.txt
```

Compared byte-identically against the prior post-refactor baseline at
`reports/routing_post_refactor_dev_2026-05-20.log` (modulo two
intrinsically-variable lines: the timestamped report path and the
embedded "Date" field):

```
diff <(grep -vE 'Report saved|routing_evaluation_2026' /tmp/routing_post_refactor.txt) \
     <(grep -vE 'Report saved|routing_evaluation_2026' reports/routing_post_refactor_dev_2026-05-20.log)
# exit 0 — files identical
```

Identical content includes:

| Slice | Result |
|---|---|
| Cardiologist (all dev cases) | 15 / 15 = 100.0% [79.6%–100.0%] |
| Endocrinologist (all dev cases) | 15 / 15 = 100.0% [79.6%–100.0%] |
| Cardiologist Tier 1 (core) | 14 / 14 = 100.0% [78.5%–100.0%] |
| Cardiologist Tier 3 (out_of_scope) | 1 / 1 = 100.0% [20.7%–100.0%] |
| Endocrinologist Tier 1 (core) | 15 / 15 = 100.0% [79.6%–100.0%] |
| Cross-domain ambiguous routing | Same 8/8 decisions, same routed-to specialties |

**Behavioural change: zero.** The dev-split routing accuracy is 100/100
identical to the pre-Stage-23 baseline (which was itself 100/100
identical to pre-Stage-8 prior to the registry consolidation — the
routing prompt has been textually stable across all three states).

## 6. Files Touched

- `multi-agent_system/orchestrator.py` — deleted `@property` block + comment
- `multi-agent_system/tests/evaluate_chunk_relevance.py` — 2 subs
- `multi-agent_system/tests/tune_refusal_gate.py` — 4 subs
- `multi-agent_system/tests/annotate_gold_sources.py` — 2 subs
- `multi-agent_system/tests/evaluate_retrieval.py` — 8 subs + 1 `.get()` patch
- `multi-agent_system/tests/inspect_judge_disagreements.py` — 2 subs
- `multi-agent_system/tests/tune_retrieval.py` — 2 subs
- `multi-agent_system/tests/evaluate_generation.py` — 4 subs
- `multi-agent_system/scratch.py` — **deleted**
- `reports/report_stage_23.md` — this stage report (new)

## 7. Not Committed

Per spec, nothing is committed. Working tree is clean for your manual
commit.

Suggested commit message (from the spec):

```
[refactor] remove backward-compat @property shims; migrate all call sites
to self.agents[key]; delete scratch.py
```
