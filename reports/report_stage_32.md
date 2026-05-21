# Stage 32 — Dead-Code and Inactive-Config Cleanup

(Filename note: next sequential number — Stages 23 through 31 are already
taken: dict-access migration, `domain_scope` routing prompt, adversarial
routing, README refresh, MRR bootstrap CI, registry schema-validation
tests, real Gradio UI test, §4.5 architectural framing, multijudge
reconciliation.)

## 1. What Was Changed (One-Line Each)

| # | Item | File | Disposition |
|---|---|---|---|
| 1 | `scratch.py` | `multi-agent_system/scratch.py` | **Already deleted** in Stage 23 (`119df519`). Verified with `test -f` (exit 1 = not present). No new action needed. |
| 2a | `CORPUS_DIST_K = -0.300` setting | `multi-agent_system/settings.py` | **Removed.** Comment block shortened from 9 lines → 3 lines per spec. `REFUSAL_GATE_SIGNAL = 'A'` and `L2_REJECT_MIN = 0.920` kept. |
| 2b | `CORPUS_DIST_K` import in production runtime | `multi-agent_system/agents/specialist.py` | **Removed.** Constructor argument changed from `corpus_dist_k=CORPUS_DIST_K` to `corpus_dist_k=1.0` (hard-coded default with inline comment noting it is inert when `REFUSAL_GATE_SIGNAL == 'A'`). |
| 2c | `_update_settings` write-back | `multi-agent_system/tests/tune_refusal_gate.py` | **Updated** — `corpus_dist_k` parameter dropped; settings block sentinel + body rewritten to match the new 3-line settings comment. Tuner still grid-searches Signal B for diagnostic completeness; the chosen `corpus_dist_k` is reported on stdout but no longer persisted. Header docstring + `--no-write-settings` help text updated to reflect this. |
| 2d | Signal B implementation in `refusal_gate.py` | `multi-agent_system/refusal_gate.py` | **Kept untouched.** The Signal B code path (`signal_b_rejects`, `corpus_dist_stats`, `corpus_dist_k`) is gated by `self.signal == 'B'` and is never reached in production (where `REFUSAL_GATE_SIGNAL == 'A'`). It already takes its parameter via constructor — exactly the shape the spec asks for ("move its parameter into the function call rather than into settings"). |
| 3 | `normalise` → `parse_or_fail` | `multi-agent_system/tests/evaluate_routing.py` | **Renamed.** 4 sites updated: `def normalise` → `def parse_or_fail` (line 67); two `normalise(raw_response)` call sites (originally lines 126 + 215); one docstring backtick reference (line 43); and the function's own docstring opening line ("Strict-equality normaliser" → "Strict-equality parser") for self-consistency, with a `— renamed from \`normalise\` in Stage 32` annotation. |
| 4 | TODO / FIXME / XXX markers | (whole codebase) | **Zero hits.** `grep -rn "TODO\|FIXME\|XXX" multi-agent_system/ tests/ scripts/` (excluding `__pycache__`) returns no matches — no triage required. |

## 2. Per-File Diff Summary

### 2.1 `multi-agent_system/settings.py`

```diff
-# --- refusal-gate constants (managed by tests/tune_refusal_gate.py) ---
-# Tuned 2026-05-19. Dev set has only one T3 case (cardio_10), so the dev FP/FN
-# binaries are too coarse to tune the threshold against the user-supplied
-# ≥80% T3 recall / ≤5% T1/T2 FP target. Threshold below was chosen as the lowest
-# value that still satisfies the test-split ≥80% T3 recall target while
-# minimizing test-split FP rate. The ≤5% T1/T2 FP target is *not* simultaneously
-# achievable on this corpus because the in-scope and out-of-scope min-L2
-# distributions overlap heavily (T3: 0.84–1.00; T1/T2: 0.70–1.07). See §4.5 of
-# report_final.md for the full trade-off curve and Stage 7 report for analysis.
+# --- refusal-gate threshold (managed by tests/tune_refusal_gate.py) ---
+# Signal A min-L2 threshold. See report_final.md §4.5 for the trade-off
+# against Tier 1/2 false-positive rate; full analysis in Stage 7 report.
 REFUSAL_GATE_SIGNAL = 'A'
 L2_REJECT_MIN = 0.920
-CORPUS_DIST_K = -0.300
```

### 2.2 `multi-agent_system/agents/specialist.py`

```diff
 from settings import (
     AGENT_MODEL,
-    CORPUS_DIST_K,
     L2_REJECT_MIN,
     MAX_L2_DISTANCE,
     REFUSAL_GATE_SIGNAL,
     SIMILARITY_TOP_K,
     YANDEX_PROJECT_ID,
     client,
 )

 ...

             self._refusal_gate = RefusalGate.from_vectorstore(
                 self.vectorstore,
                 specialty=specialty,
                 processed_dir=self.folder_path,
                 l2_reject_min=L2_REJECT_MIN,
-                corpus_dist_k=CORPUS_DIST_K,
+                corpus_dist_k=1.0,  # Signal B parameter; inert when REFUSAL_GATE_SIGNAL=='A'.
                 signal=REFUSAL_GATE_SIGNAL,
                 top_k=SIMILARITY_TOP_K,
             )
```

### 2.3 `multi-agent_system/tests/tune_refusal_gate.py`

```diff
   - settings.py (updated)              — writes REFUSAL_GATE_SIGNAL and L2_REJECT_MIN (Signal A). Signal B's chosen `corpus_dist_k` is reported on stdout but no longer persisted to settings (post-Stage-32 cleanup); pass it into `RefusalGate(corpus_dist_k=...)` at construction time.

 ...

     parser.add_argument("--no-write-settings", action="store_true",
-                        help="Skip writing L2_REJECT_MIN and CORPUS_DIST_K back to settings.py")
+                        help="Skip writing REFUSAL_GATE_SIGNAL and L2_REJECT_MIN back to settings.py")

 ...

     if not args.no_write_settings:
         _update_settings(chosen_signal=chosen_signal,
-                         l2_reject_min=chosen_l2,
-                         corpus_dist_k=chosen_k)
+                         l2_reject_min=chosen_l2)
         print(f"\nUpdated {SETTINGS_PATH}:")
         print(f"  REFUSAL_GATE_SIGNAL    = {chosen_signal!r}")
         print(f"  L2_REJECT_MIN          = {chosen_l2}")
-        print(f"  CORPUS_DIST_K          = {chosen_k}")
+        if chosen_signal == 'B':
+            print(f"  (Signal B chosen with CORPUS_DIST_K={chosen_k:.3f}; "
+                  "this value is reported here but no longer persisted to "
+                  "settings.py — pass it into RefusalGate(corpus_dist_k=...) "
+                  "at construction time. See refusal_gate.py.)")
     else:
         print("\n(skipping settings.py write because --no-write-settings was passed)")


-def _update_settings(*, chosen_signal: str, l2_reject_min: float, corpus_dist_k: float):
-    """Append/replace REFUSAL_GATE_SIGNAL, L2_REJECT_MIN, CORPUS_DIST_K in settings.py."""
+def _update_settings(*, chosen_signal: str, l2_reject_min: float):
+    """Append/replace REFUSAL_GATE_SIGNAL and L2_REJECT_MIN in settings.py.
+
+    CORPUS_DIST_K is intentionally no longer written here — the Stage 32
+    cleanup removed it from settings.py because Signal A is the production
+    runtime path. If a future Signal-B re-tune chooses a non-default
+    `corpus_dist_k`, pass it into `RefusalGate(corpus_dist_k=...)` at
+    construction time rather than re-adding the settings constant.
+    """
     with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
         text = f.read()
-    sentinel = "# --- refusal-gate constants (managed by tests/tune_refusal_gate.py) ---"
+    sentinel = "# --- refusal-gate threshold (managed by tests/tune_refusal_gate.py) ---"
     block = (
         f"\n{sentinel}\n"
+        f"# Signal A min-L2 threshold. See report_final.md §4.5 for the trade-off\n"
+        f"# against Tier 1/2 false-positive rate; full analysis in Stage 7 report.\n"
         f"REFUSAL_GATE_SIGNAL = {chosen_signal!r}\n"
         f"L2_REJECT_MIN = {float(l2_reject_min):.3f}\n"
-        f"CORPUS_DIST_K = {float(corpus_dist_k):.3f}\n"
     )
```

### 2.4 `multi-agent_system/tests/evaluate_routing.py`

```diff
-    returned as-is; `normalise()` validates it against `ALLOWED_SPECIALISTS`.
+    returned as-is; `parse_or_fail()` validates it against `ALLOWED_SPECIALISTS`.

 ...

-def normalise(raw: str) -> str:
-    """Strict-equality normaliser (Stage 19).
+def parse_or_fail(raw: str) -> str:
+    """Strict-equality parser (Stage 19) — renamed from `normalise` in Stage 32.

 ...
-        predicted = normalise(raw_response)
+        predicted = parse_or_fail(raw_response)
     ... (×2 call sites)
```

## 3. Smoke-Test Outputs (verbatim per spec)

```
--- scratch.py absent (expect EXIT:1) ---
EXIT:1
--- CORPUS_DIST_K absent from settings.py (expect no match) ---
EXIT:1
--- L2_REJECT_MIN = 0.920 present (expect >=1) ---
1
--- def normalise absent (expect no match) ---
EXIT:1
--- def parse_or_fail present (expect >=1) ---
1
```

All five spec smoke tests pass.

## 4. Pytest Output

```
$ python -m pytest tests/ -q
... (8 upstream-dep DeprecationWarnings from Gradio/Pandas — pre-existing)
40 passed, 1 skipped, 8 warnings in 6.19s
```

**40 passed, 1 skipped** — same pass count as the post-Stage-29 baseline.
No test exercises `CORPUS_DIST_K` or the old `normalise()` name (confirmed
by Stage 28's full grep + the fact that the test count is unchanged).

## 5. Dev Routing Match

```
$ cd multi-agent_system
$ python tests/evaluate_routing.py --split dev
... (full per-case log)
  cardiologist               15       15  100.0% [79.6%–100.0%]
  endocrinologist            15       15  100.0% [79.6%–100.0%]
  OVERALL                    30       30  100.0% [88.6%–100.0%]
  cardiologist         1      core               14      14  100.0% [78.5%–100.0%]
  cardiologist         3      out_of_scope        1       1  100.0% [20.7%–100.0%]
```

**30/30 = 100.0% [88.6%–100.0%]** — exactly matches the post-Stage-24
dev routing baseline. The rename `normalise → parse_or_fail` is a pure
identifier change with no behavioural side effect.

## 6. TODO / FIXME / XXX Triage

Command (verbatim per spec):

```
grep -rn "TODO\|FIXME\|XXX" multi-agent_system/ tests/ scripts/
```

Output: **(empty — exit code 1, no matches)**.

The codebase carries zero in-comment `TODO` / `FIXME` / `XXX` markers
across the three target trees (excluding `__pycache__`). No triage
required. The Stage 17 six-defect cleanup precedent (where every
defect was either resolved inline or recorded in `report_final.md` §6
Limitations) has been followed forward: subsequent stages have similarly
either resolved or surfaced open items into the `report_final.md` /
stage report path, never via stray `TODO` comments.

## 7. Files Touched

- `multi-agent_system/settings.py` — 9-line comment + 1 setting line removed; replaced with 3-line comment + 2 settings kept
- `multi-agent_system/agents/specialist.py` — 1 import dropped; 1 constructor argument changed
- `multi-agent_system/tests/tune_refusal_gate.py` — header docstring, CLI help, `_update_settings()` signature + body, call site
- `multi-agent_system/tests/evaluate_routing.py` — 4 `normalise` → `parse_or_fail` renames + docstring annotation
- `reports/report_stage_32.md` — this stage report (new)
