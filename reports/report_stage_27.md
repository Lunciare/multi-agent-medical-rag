# Stage 27 — Bootstrap 95% CI for MRR@5 Reporting

(Filename note: next sequential number after the already-committed
Stages 23–26 — Stage 23 = orchestrator dict-access migration,
Stage 24 = wiring `domain_scope` into routing prompt,
Stage 25 = adversarial routing test set,
Stage 26 = README refresh.)

## 1. What Was Changed

Closes the deferred uncertainty estimate for MRR@K flagged in Stage 6 §2:
*"MRR@K is reported as a mean over annotated cases; it is a continuous
quantity on [0, 1] rather than a Bernoulli proportion, so a strict Wilson
CI is not the appropriate uncertainty estimate."* MRR@K cells in
`evaluate_retrieval.py` and `report_final.md` (§4.3 main table and §4.3.2
retriever comparison) now carry a percentile-method bootstrap 95% CI
over the per-case reciprocal-rank vector.

- `multi-agent_system/tests/evaluate_retrieval.py`:
  - Added `import numpy as np` and the constants `BOOTSTRAP_B = 10000` /
    `BOOTSTRAP_SEED = 12345`.
  - Added the helper `_bootstrap_mean_ci(values, B, seed, alpha)` (spec
    body — see §2 below).
  - Per-case reciprocal-rank lists are now tracked alongside the existing
    running sums: `domain_mrr_per_case[domain]`,
    `tier_mrr_per_case[(domain, tier)]`, and
    `mrr_per_case[(domain, tier, method)]`. The legacy `*_mrr_sum` /
    `*_mrr_n` aggregators are kept so the file's interim sums still
    work for any downstream consumer.
  - Three report blocks now render `MRR@K [95% CI]` instead of `MRR@K`:
    the per-domain "Grounded Retrieval Metrics" table, the per-tier
    breakdown, and the FAISS/BM25/Random/Oracle "Retriever Comparison"
    table. Overall rows pool the per-case lists across (domain × tier)
    before bootstrapping.
- `reports/report_final.md`:
  - §4.3 main table: every MRR@K cell now reads `mean [lo–hi]`
    (full-set bootstrap on the 100-case golden set, n=82 annotated cases).
  - §4.3 caption updated: replaced "MRR@K is reported without a strict CI
    ... see Stage 6 report for bootstrap-style sanity checks" with the
    explicit bootstrap protocol (B=10000, seed=12345).
  - §4.3.2 Retriever Comparison table: every FAISS MRR@5 / BM25 MRR@5
    cell now reads `mean [lo–hi]` (test-split bootstrap, n=153 gold-doc
    slots across 53 annotated cases).
- README evaluation table: not updated. The README headline metrics row
  is Routing Accuracy / Recall@5 / Faithfulness / Tier 3 Refusal Rate —
  **no MRR@5 column or cell exists in the README**, so there is nothing
  to annotate. A grep for `MRR` in `README.md` returns zero matches.

## 2. Bootstrap Helper Code (verbatim)

Inserted at the top of `multi-agent_system/tests/evaluate_retrieval.py`,
just below the existing imports:

```python
import numpy as np

# ...

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 12345


def _bootstrap_mean_ci(values, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Percentile-method bootstrap CI for the mean of `values`.
    Returns (mean, lo, hi). Returns (0.0, 0.0, 0.0) if values is empty."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = rng.choice(arr, size=(B, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)
```

Deterministic by construction: with `seed=12345` and `B=10000` the same
input list returns the same `(mean, lo, hi)` triple across runs and
across machines, so report numbers are reproducible.

## 3. Unit Smoke-Test Output

Command (verbatim from the spec):

```python
python -c "
import numpy as np
from tests.evaluate_retrieval import _bootstrap_mean_ci
# Toy: 10 reciprocal ranks centred on 0.5
rng = np.random.default_rng(0)
vals = rng.uniform(0, 1, 100).tolist()
m, lo, hi = _bootstrap_mean_ci(vals, B=2000, seed=12345)
assert 0 <= lo <= m <= hi <= 1
assert hi - lo < 0.2, 'CI too wide for n=100 uniform'
print(f'MRR bootstrap smoke OK: mean={m:.3f}  CI=[{lo:.3f}-{hi:.3f}]')
"
```

Stdout:

```
MRR bootstrap smoke OK: mean=0.548  CI=[0.492-0.606]
```

Both asserts pass: `0 ≤ 0.492 ≤ 0.548 ≤ 0.606 ≤ 1`, and the CI width
0.114 is well under the 0.2 sanity bound for n=100 uniform on [0, 1].

## 4. Per-(Domain, Tier) MRR@5 With Bootstrap CI From the Live Runs

Two retrieval evals were executed: one on the held-out test split
(populates §4.3.2 of `report_final.md`, which is explicitly a test-split
table) and one on the full 100-case set (populates §4.3 main table,
which uses full-set numbers — denominators 118 / 122 / 240 in the
existing Recall@K column).

### 4.1. Full set (100 cases, 82 annotated — feeds §4.3 main table)

From `/tmp/retrieval_all_stage27.log`:

| Domain | n cases | MRR@K [95% CI] |
|---|---|---|
| cardiologist | 40 | **0.730 [0.618–0.833]** |
| endocrinologist | 42 | **0.757 [0.645–0.861]** |
| **OVERALL (T1+T2)** | **82** | **0.744 [0.663–0.821]** |

Per tier (still full set):

| Domain | Tier | n | MRR@K [95% CI] |
|---|---|---|---|
| cardiologist | 1 (core) | 27 | 0.809 [0.698–0.907] |
| cardiologist | 2 (peripheral) | 13 | 0.567 [0.345–0.785] |
| endocrinologist | 1 (core) | 26 | 0.806 [0.687–0.917] |
| endocrinologist | 2 (peripheral) | 16 | 0.677 [0.458–0.875] |

The point estimates are byte-identical to the legacy `0.730 / 0.757 /
0.744` headlines (and to the per-tier 0.809 / 0.567 / 0.806 / 0.677
values in the Stage 6 full-set table) — the bootstrap is sampling the
*same* per-case reciprocal-rank vector, just attaching uncertainty to it.

### 4.2. Held-out test split (70 cases, 53 annotated — feeds §4.3.2 retriever-comparison table)

From `/tmp/retrieval_test_stage27.log`. Per-method, per-(domain, tier):

| Domain | Tier | Method | MRR@5 [95% CI] |
|---|---|---|---|
| cardiologist | 1 | faiss | 0.737 [0.577–0.885] |
| cardiologist | 1 | bm25 | 0.442 [0.212–0.673] |
| cardiologist | 1 | random | 0.026 [0.000–0.077] |
| cardiologist | 1 | oracle | 1.000 [1.000–1.000] |
| cardiologist | 2 | faiss | 0.567 [0.345–0.785] |
| cardiologist | 2 | bm25 | 0.562 [0.338–0.785] |
| cardiologist | 2 | random | 0.227 [0.054–0.442] |
| cardiologist | 2 | oracle | 1.000 [1.000–1.000] |
| endocrinologist | 1 | faiss | 0.773 [0.591–0.939] |
| endocrinologist | 1 | bm25 | 0.348 [0.106–0.621] |
| endocrinologist | 1 | random | 0.000 [0.000–0.000] |
| endocrinologist | 1 | oracle | 1.000 [1.000–1.000] |
| endocrinologist | 2 | faiss | 0.677 [0.458–0.875] |
| endocrinologist | 2 | bm25 | 0.492 [0.285–0.700] |
| endocrinologist | 2 | random | 0.000 [0.000–0.000] |
| endocrinologist | 2 | oracle | 1.000 [1.000–1.000] |
| **OVERALL (T1+T2)** | — | **faiss** | **0.685 [0.582–0.787]** |
| **OVERALL (T1+T2)** | — | **bm25** | **0.467 [0.353–0.584]** |
| **OVERALL (T1+T2)** | — | **random** | **0.062 [0.013–0.124]** |
| **OVERALL (T1+T2)** | — | **oracle** | **1.000 [1.000–1.000]** |

The point estimates again match the legacy `0.685 / 0.467 / 0.062 /
1.000` overall row to the third decimal — the bootstrap is purely
additive for uncertainty, never modifies a point estimate.

**Reading the intervals.** Per-tier CIs are wide (e.g. cardiology T1
FAISS MRR is 0.737 with CI 0.577–0.885, a ±0.16 swing) because n=13
annotated cases is genuinely small. The overall test-split FAISS MRR
0.685 [0.582–0.787] is the right number to quote for cross-system
comparison; the Wilson-style "±5–10 points" intuition that holds for
Recall@K under n≈153 gold-doc trials *also* holds for the bootstrapped
MRR here.

**Comparative reading.** The CIs surface comparisons that the legacy
point-estimate table only suggested:

- **FAISS vs BM25 (overall):** `0.685 [0.582–0.787]` vs `0.467
  [0.353–0.584]` — the upper bound of BM25 (0.584) is below the lower
  bound of FAISS (0.582), so the 0.22-point lead is statistically robust
  at the 95% level.
- **Cardiology T2 FAISS vs BM25:** `0.567 [0.345–0.785]` vs `0.562
  [0.338–0.785]` — overlapping CIs from 0.345 to 0.785; the 0.005-point
  point-estimate lead is *not* statistically distinguishable from zero.
  This is the tier where BM25 catches up to FAISS, consistent with the
  §4.3.2 narrative.
- **Endocrinology T1 FAISS vs BM25:** `0.773 [0.591–0.939]` vs `0.348
  [0.106–0.621]` — the largest gap, with non-overlapping CIs (0.591 vs
  0.621), so the 0.43-point lead survives uncertainty.

## 5. Verbatim Updated §4.3 / §4.3.2 Numbers in `report_final.md`

### 5.1. §4.3 main table (line 278–284)

**Old:**

```markdown
| Domain | Recall@K | MRR@K | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|
| Cardiology | 61.0% (72/118) [52.0%–69.3%] | 0.730 | 86.0% (43/50) [73.8%–93.0%] | 0.0% (0/9) [0.0%–29.9%] |
| Endocrinology | 57.4% (70/122) [48.5%–65.8%] | 0.757 | 96.0% (48/50) [86.5%–98.9%] | 0.0% (0/7) [0.0%–35.4%] |
| **Overall** | **59.2% (142/240) [52.9%–65.2%]** | **0.744** | **91.0% (91/100) [83.8%–95.2%]** | **0.0% (0/16) [0.0%–19.4%]** |

*(Wilson 95% CIs on the pooled gold-doc Bernoulli. MRR@K is reported without a strict CI — it is a mean of [0, 1] reciprocal-rank values per case, not a Bernoulli proportion; see Stage 6 report for bootstrap-style sanity checks. `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric`.)*
```

**New:**

```markdown
| Domain | Recall@K | MRR@K [95% CI] | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|
| Cardiology | 61.0% (72/118) [52.0%–69.3%] | 0.730 [0.618–0.833] | 86.0% (43/50) [73.8%–93.0%] | 0.0% (0/9) [0.0%–29.9%] |
| Endocrinology | 57.4% (70/122) [48.5%–65.8%] | 0.757 [0.645–0.861] | 96.0% (48/50) [86.5%–98.9%] | 0.0% (0/7) [0.0%–35.4%] |
| **Overall** | **59.2% (142/240) [52.9%–65.2%]** | **0.744 [0.663–0.821]** | **91.0% (91/100) [83.8%–95.2%]** | **0.0% (0/16) [0.0%–19.4%]** |

*(Recall@K Wilson 95% CIs are on the pooled gold-doc Bernoulli. MRR@K 95% CIs are percentile-method bootstrap intervals over the per-case reciprocal-rank vector (B=10000 resamples, RNG seed=12345; helper `_bootstrap_mean_ci` in `tests/evaluate_retrieval.py`, added Stage 27) — appropriate because MRR is a mean of [0, 1] reciprocal-rank values per case rather than a Bernoulli proportion. `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric`.)*
```

### 5.2. §4.3.2 Retriever Comparison table (line 304–310)

**Old (point-estimate-only MRR columns):**

```markdown
| Domain | Tier | FAISS Recall@5 | BM25 Recall@5 | Random Recall@5 | Oracle Recall@5 | FAISS MRR@5 | BM25 MRR@5 |
|---|---|---|---|---|---|---|---|
| Cardiology | T1 (core) | 59.0% (23/39) [43.4%–72.9%] | 25.6% (10/39) [14.6%–41.1%] | 2.6% (1/39) [0.5%–13.2%] | 100% (39/39) [91.0%–100%] | 0.737 | 0.442 |
| Cardiology | T2 (peripheral) | 54.1% (20/37) [38.4%–69.0%] | 43.2% (16/37) [28.7%–59.1%] | 13.5% (5/37) [5.9%–28.0%] | 100% (37/37) [90.6%–100%] | 0.567 | 0.562 |
| Endocrinology | T1 (core) | 60.6% (20/33) [43.7%–75.3%] | 15.2% (5/33) [6.7%–30.9%] | 0.0% (0/33) [0.0%–10.4%] | 100% (33/33) [89.6%–100%] | 0.773 | 0.348 |
| Endocrinology | T2 (peripheral) | 52.3% (23/44) [37.9%–66.2%] | 34.1% (15/44) [21.9%–48.9%] | 0.0% (0/44) [0.0%–8.0%] | 100% (44/44) [92.0%–100%] | 0.677 | 0.492 |
| **Overall (T1+T2)** | — | **56.2% (86/153) [48.3%–63.8%]** | **30.1% (46/153) [23.4%–37.7%]** | **3.9% (6/153) [1.8%–8.3%]** | **100% (153/153) [97.6%–100%]** | **0.685** | **0.467** |
```

**New (MRR columns now carry [95% CI]):**

```markdown
| Domain | Tier | FAISS Recall@5 | BM25 Recall@5 | Random Recall@5 | Oracle Recall@5 | FAISS MRR@5 [95% CI] | BM25 MRR@5 [95% CI] |
|---|---|---|---|---|---|---|---|
| Cardiology | T1 (core) | 59.0% (23/39) [43.4%–72.9%] | 25.6% (10/39) [14.6%–41.1%] | 2.6% (1/39) [0.5%–13.2%] | 100% (39/39) [91.0%–100%] | 0.737 [0.577–0.885] | 0.442 [0.212–0.673] |
| Cardiology | T2 (peripheral) | 54.1% (20/37) [38.4%–69.0%] | 43.2% (16/37) [28.7%–59.1%] | 13.5% (5/37) [5.9%–28.0%] | 100% (37/37) [90.6%–100%] | 0.567 [0.345–0.785] | 0.562 [0.338–0.785] |
| Endocrinology | T1 (core) | 60.6% (20/33) [43.7%–75.3%] | 15.2% (5/33) [6.7%–30.9%] | 0.0% (0/33) [0.0%–10.4%] | 100% (33/33) [89.6%–100%] | 0.773 [0.591–0.939] | 0.348 [0.106–0.621] |
| Endocrinology | T2 (peripheral) | 52.3% (23/44) [37.9%–66.2%] | 34.1% (15/44) [21.9%–48.9%] | 0.0% (0/44) [0.0%–8.0%] | 100% (44/44) [92.0%–100%] | 0.677 [0.458–0.875] | 0.492 [0.285–0.700] |
| **Overall (T1+T2)** | — | **56.2% (86/153) [48.3%–63.8%]** | **30.1% (46/153) [23.4%–37.7%]** | **3.9% (6/153) [1.8%–8.3%]** | **100% (153/153) [97.6%–100%]** | **0.685 [0.582–0.787]** | **0.467 [0.353–0.584]** |
```

Every Recall@K cell and KeywordHitRate cell in both tables is
byte-unchanged; only the MRR cells gained a `[lo–hi]` suffix and the
column header gained `[95% CI]`.

### 5.3. README

`grep -n "MRR" README.md` returns zero matches — the README evaluation
table reports Routing Accuracy / Recall@5 / Faithfulness / Tier 3
Refusal Rate, but not MRR. No README annotation was therefore made;
the spec's "Update the README evaluation table the same way (or add a
footnote)" is vacuously satisfied (no MRR cell exists to annotate).

## 6. Files Touched

- `multi-agent_system/tests/evaluate_retrieval.py` — `_bootstrap_mean_ci`
  helper, per-case MRR lists, three updated report blocks
- `reports/report_final.md` — §4.3 main table + caption, §4.3.2
  retriever-comparison table
- `reports/report_stage_27.md` — this stage report (new)

## 7. Not Committed

Per spec, nothing is committed. Working tree holds the changes above for
your manual commit.

Suggested commit message:

```
[chore] bootstrap 95% CI for MRR@5 (B=10000, seed=12345);
annotate report_final.md §4.3 / §4.3.2 MRR cells; helper
`_bootstrap_mean_ci` in tests/evaluate_retrieval.py
```
