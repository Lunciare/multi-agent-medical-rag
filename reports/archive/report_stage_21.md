# Stage 21 — Landis & Koch Marginal-Degeneracy Guard in `evaluate_generation.py`

## 1. What Was Changed

`_landis_koch(kappa)` previously returned a flat Landis & Koch label
("poor", "moderate", "almost perfect", …) regardless of whether κ was a
meaningful agreement statistic or a degenerate artifact of a one-sided
marginal. Stage 5 §4 already documented the degeneracy in prose
(primary judge marks every test case FAITHFUL → P(HALLUCINATION) = 0 →
expected agreement equals observed agreement → κ collapses to exactly 0),
but the generated markdown rendered this as a misleading **"poor"**
agreement label. The 2026-05-20 11:06:59 run was even worse: both judges
returned 100% FAITHFUL, `cohen_kappa_score` returned NaN, and the old
function fell through the sign/range cascade (every `nan < x` is False) and
rendered **"almost perfect"** — the most flattering possible label.

This stage replaces the silent fall-through with two explicit
degeneracy branches:

1. **Marginal degeneracy** — if either judge emitted fewer than two
   distinct labels across the run, the function returns
   `"degenerate (one marginal = 0; observed agreement = A/N)"`. This fires
   even when κ is a numeric `0.000`, because the κ value itself is
   structurally constrained, not informative.
2. **Undefined κ** — if `cohen_kappa_score` returned `None` or NaN, the
   function returns `"degenerate (κ undefined)"`.

Per-judge `label_counts` are built once in `_write_outputs` before the
pair-iteration loop and passed by keyword into `_landis_koch`.

## 2. Diff (`multi-agent_system/tests/evaluate_generation.py`)

```diff
diff --git a/multi-agent_system/tests/evaluate_generation.py b/multi-agent_system/tests/evaluate_generation.py
index f1f58e0d..2c2a166b 100644
--- a/multi-agent_system/tests/evaluate_generation.py
+++ b/multi-agent_system/tests/evaluate_generation.py
@@ -1,5 +1,6 @@
 import argparse
 import csv
+import math
 import os
 import sys
 import json
@@ -310,13 +311,20 @@ def _write_outputs(judges, per_case_rows, stats_by_judge, split,
         "| Pair | n (both non-None) | Agreements | Cohen's κ | Landis & Koch |",
         "|---|---|---|---|---|",
     ]
+    label_counts = {j.name: {True: 0, False: 0} for j in judges}
+    for row in judged_rows:
+        for j in judges:
+            v = _judge_label_to_bool(row[j.name])
+            if v is not None:
+                label_counts[j.name][v] += 1
     for pk in pair_kappas:
-        if pk["kappa"] is None:
-            kappa_str = "n/a"
-            lk = "n/a"
-        else:
-            kappa_str = f"{pk['kappa']:.3f}"
-            lk = _landis_koch(pk["kappa"])
+        kappa_str = "n/a" if pk["kappa"] is None else f"{pk['kappa']:.3f}"
+        lk = _landis_koch(
+            pk["kappa"],
+            both_labels_seen_a=sum(1 for v, c in label_counts[pk["a"]].items() if c > 0),
+            both_labels_seen_b=sum(1 for v, c in label_counts[pk["b"]].items() if c > 0),
+            n=pk["n"], agreements=pk["agreements"],
+        )
         lines.append(
             f"| ({pk['a']}, {pk['b']}) | {pk['n']} | {pk['agreements']} | {kappa_str} | {lk} |"
         )
@@ -368,7 +376,12 @@ def _write_outputs(judges, per_case_rows, stats_by_judge, split,
           f"total judge calls: {total_judge_calls}")


-def _landis_koch(kappa: float) -> str:
+def _landis_koch(kappa: float, *, both_labels_seen_a: int, both_labels_seen_b: int,
+                 n: int, agreements: int) -> str:
+    if both_labels_seen_a < 2 or both_labels_seen_b < 2:
+        return f"degenerate (one marginal = 0; observed agreement = {agreements}/{n})"
+    if kappa is None or math.isnan(kappa):
+        return "degenerate (κ undefined)"
     if kappa < 0.0:
         return "less than chance"
     if kappa < 0.4:
```

## 3. Unit Smoke-Test Output

Command (verbatim from the spec):

```
cd multi-agent_system && python -c "
import math
from tests.evaluate_generation import _landis_koch
# Marginal-degenerate case (primary always FAITHFUL):
assert _landis_koch(0.0, both_labels_seen_a=1, both_labels_seen_b=2,
                    n=70, agreements=69).startswith('degenerate'), 'failed degeneracy'
# Normal case:
assert _landis_koch(0.55, both_labels_seen_a=2, both_labels_seen_b=2,
                    n=70, agreements=60) == 'moderate', 'failed normal'
# nan case:
assert _landis_koch(float('nan'), both_labels_seen_a=2, both_labels_seen_b=2,
                    n=58, agreements=58).startswith('degenerate'), 'failed nan'
print('Landis & Koch degeneracy guard OK')
"
```

Stdout:

```
Landis & Koch degeneracy guard OK
```

All three asserts (degeneracy / normal / nan) passed. PASS.

## 4. Re-Run Multi-Judge Eval: New (yandex_primary, secondary) Label

Command:

```
cd multi-agent_system && SECONDARY_JUDGE_PROVIDER="yandex:gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest" \
  python tests/evaluate_generation.py --split test --mode multi_judge
```

(The `SECONDARY_JUDGE_PROVIDER` env var was set inline for the run — it is
not yet persisted in `.env`. The URI is identical to the one used in the
2026-05-19 multi-judge run.)

Re-run summary (from `reports/faithfulness_multijudge_2026-05-20.md`,
written 2026-05-20 20:53:17 UTC, 358.8s, 116 judge calls):

```
| Pair                          | n  | Agreements | Cohen's κ | Landis & Koch                                                |
|-------------------------------|----|------------|-----------|--------------------------------------------------------------|
| (yandex_primary, secondary)   | 58 |    57      |  0.000    | degenerate (one marginal = 0; observed agreement = 57/58)    |
```

The Landis & Koch column reads exactly:

> `degenerate (one marginal = 0; observed agreement = 57/58)`

— the spec's "or similar" target. The label now surfaces the structural
problem (primary's `P(HALLUCINATION) = 0`) instead of mislabelling the
constraint as "poor" agreement.

## 5. Confirmation: No Faithfulness Rates Changed

The diff in §2 touches only:

- `_landis_koch` (string-label rendering function); and
- a `label_counts` construction in `_write_outputs` that is **read-only**
  with respect to every faithfulness statistic — the counts are consumed
  exclusively by the new keyword args to `_landis_koch`.

No code path that computes a per-judge faithful count, minimum-judge
faithful count, Wilson CI, or fallback total was modified. The per-judge
faithful tallies are accumulated in the loop at lines 217-227 of
`evaluate_generation.py`; the minimum-judge tally is accumulated at lines
257-262. Both blocks are byte-identical pre- and post-change.

Numerically, comparing the re-run summary above to the prior multi-judge
run on the same test split (`reports/faithfulness_multijudge_2026-05-19.md`):

| Metric                                  | 2026-05-19 (old code) | 2026-05-20 re-run (new code) |
|-----------------------------------------|------------------------|------------------------------|
| yandex_primary FAITHFUL                 | 58/58 = 100.0%         | 58/58 = 100.0%               |
| secondary FAITHFUL                      | 57/58 = 98.3%          | 57/58 = 98.3%                |
| Minimum-judge FAITHFUL                  | 57/58 = 98.3%          | 57/58 = 98.3%                |
| Pair n / agreements / κ                 | 58 / 57 / 0.000        | 58 / 57 / 0.000              |
| Disagreement case                       | `cardio_40`            | `cardio_40`                  |

Every rate, count, Wilson CI, and the κ value are identical. The only
change in the rendered markdown is the Landis & Koch column moving from
`poor` to `degenerate (one marginal = 0; observed agreement = 57/58)`.

(Note: the earlier 2026-05-20 11:06:59 file showed an outlier where the
secondary judge happened to mark `cardio_40` FAITHFUL — pushing secondary
to 58/58 and forcing `cohen_kappa_score` to return NaN. That divergence is
LLM-temperature noise between runs of the **same** code, not a code-change
artifact. Under the new code, that NaN would now render as
`degenerate (κ undefined)` instead of the previous misleading
`almost perfect`.)

## 6. Files Touched

- `multi-agent_system/tests/evaluate_generation.py` — degeneracy guard + label_counts plumbing
- `reports/faithfulness_multijudge_2026-05-20.md` — regenerated by the re-run (overwrote the earlier 2026-05-20 run)
- `reports/faithfulness_multijudge_raw_2026-05-20.csv` — regenerated by the re-run
- `reports/report_stage_21.md` — this stage report (new) s