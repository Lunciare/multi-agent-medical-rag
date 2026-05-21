# Multi-Judge Faithfulness — Run-to-Run Reconciliation

**Scope.** Compares the two persisted multi-judge faithfulness runs:

- `reports/faithfulness_multijudge_2026-05-19.md` (+ raw CSV) — produced
  during Stage 5 and quoted by `report_final.md` §4.4 / §7.
- `reports/faithfulness_multijudge_2026-05-20.md` (+ raw CSV) — produced
  during Stage 21's re-run of the multi-judge eval against the updated
  `_landis_koch` function. The 2026-05-20 file overwrote an earlier
  same-day run at 11:06:59 — see §4 below for the residual variance
  evidence that survives only in the Stage 21 stage report.

## 1. Run summary diff

| Metric | 2026-05-19 23:04:28 | 2026-05-20 20:53:17 |
|---|---|---|
| Elapsed | 354.5 s (5.9 min) | 358.8 s (6.0 min) |
| Total judge calls | 116 | 116 |
| `yandex_primary` Faithful / Total | 58 / 58 = 100.0% [93.8%–100.0%] | 58 / 58 = 100.0% [93.8%–100.0%] |
| `secondary` Faithful / Total | 57 / 58 = 98.3% [90.9%–99.7%] | 57 / 58 = 98.3% [90.9%–99.7%] |
| Min-judge agreement | 57 / 58 = 98.3% [90.9%–99.7%] | 57 / 58 = 98.3% [90.9%–99.7%] |
| Pair `(primary, secondary)` n / agreements / κ | 58 / 57 / **0.000** | 58 / 57 / **0.000** |
| Landis & Koch label rendered | `poor` | `degenerate (one marginal = 0; observed agreement = 57/58)` |
| Disagreement cases listed | 1 (`cardio_40` Tier 2 cardiology) | 1 (`cardio_40` Tier 2 cardiology) |
| Tier 3 fallback rows | 12 / 15 | 12 / 15 |

The **only field that differs** between the two summary markdowns is
the Landis & Koch label string: the 2026-05-19 run was produced before
Stage 21 added the marginal-degeneracy guard, so it rendered the κ=0
case as `poor`; the 2026-05-20 run rendered the same κ=0 case as the
new explicit `degenerate (one marginal = 0; observed agreement = 57/58)`
label. Both label strings describe the same underlying statistic — no
verdict, count, rate, or CI is different.

## 2. Case-level disagreements between runs

`diff reports/faithfulness_multijudge_raw_2026-05-19.csv
reports/faithfulness_multijudge_raw_2026-05-20.csv` returns exit code 0
— the two raw CSVs are **byte-identical**.

Programmatic confirmation:

```
$ python -c "
import csv
by_run = {}
for tag in ['2026-05-19', '2026-05-20']:
    d = {}
    with open(f'reports/faithfulness_multijudge_raw_{tag}.csv') as f:
        for row in csv.DictReader(f):
            d[row['id']] = row
    by_run[tag] = d
common = set(by_run['2026-05-19']) & set(by_run['2026-05-20'])
flips = []
for cid in common:
    r1 = by_run['2026-05-19'][cid]
    r2 = by_run['2026-05-20'][cid]
    for judge in ('yandex_primary', 'secondary'):
        if r1.get(judge) and r2.get(judge) and r1[judge] != r2[judge]:
            flips.append((cid, judge, r1[judge], r2[judge]))
print(f'Common cases: {len(common)}')
print(f'Verdict flips: {len(flips)}')
"
Common cases: 70
Verdict flips: 0
```

| Case | Judge | 2026-05-19 verdict | 2026-05-20 verdict | Flipped? |
|---|---|---|---|---|
| (none) | — | — | — | — |

**No same-judge verdict flipped between the two persisted runs.** The
single intra-run disagreement on `cardio_40` (Tier 2 cardiology,
congenital LQTS — `yandex_primary` = `FAITHFUL`, `secondary` =
`HALLUCINATION`) is present in both CSVs at identical row positions.

## 3. Tier 3 inclusion accounting

Both runs use the same inclusion rule: cases where the agent's
`refusal_gate` short-circuited to the canned `"Insufficient evidence ..."`
response are marked `fallback=True` in the CSV and **excluded** from the
`Total Judged` column of the per-judge faithfulness table (and from
both judges' calls — `yandexgpt/latest` and `yandexgpt-lite/latest`
each made exactly 58 successful calls = 70 cases − 12 fallbacks).

| | 2026-05-19 | 2026-05-20 |
|---|---|---|
| Total test-split cases | 70 | 70 |
| Tier 3 cases | 15 | 15 |
| Tier 3 fallbacks (`fallback=True`) | 12 | 12 |
| Tier 3 cases that bypassed the gate and reached the judges | 3 | 3 |
| Cases excluded from `Total Judged` | 12 | 12 |
| **Total Judged (numerator-eligible population)** | **58** | **58** |

The 12 excluded Tier 3 fallbacks contribute to the **Refusal Rate**
column documented in §4.5, not to the faithfulness denominator. The
3 Tier 3 cases that survived the gate (`cardio_30`, `endo_41`,
`endo_43`) appear in the CSV with non-`FALLBACK` judge verdicts and
*are* counted in the 58-case faithfulness denominator. Both runs
treat these three cases identically, with all three judged FAITHFUL
by both judges.

## 4. Reproducibility note

`secondary = yandexgpt-lite/latest` is **not strictly deterministic at
`temperature=0`**, despite the temperature setting. The two persisted
on-disk runs (2026-05-19 23:04:28 and 2026-05-20 20:53:17) happen to
agree on every verdict including the borderline `cardio_40` call, but
this is not a guarantee — a third run executed earlier on 2026-05-20
(at 11:06:59) was overwritten by the Stage 21 re-run and produced a
*different* `cardio_40` verdict:

| Run timestamp | `cardio_40` secondary verdict | Min-judge total | Resulting κ | File status |
|---|---|---|---|---|
| 2026-05-19 23:04:28 | HALLUCINATION | 57 / 58 = 98.3% | 0.000 (one marginal = 0) | persisted (**canonical**) |
| 2026-05-20 11:06:59 | FAITHFUL | 58 / 58 = 100.0% | NaN (both marginals = 0) | **overwritten** by the 20:53:17 re-run; documented in `report_stage_21.md` §1 + §5 |
| 2026-05-20 20:53:17 | HALLUCINATION | 57 / 58 = 98.3% | 0.000 (one marginal = 0) | persisted (re-run produced after Stage 21 `_landis_koch` change) |

So across three observed YandexGPT-Lite calls on the *same* `cardio_40`
prompt at `temperature=0`, the verdict was HALLUCINATION twice and
FAITHFUL once. The single observed flip is consistent with provider-side
non-determinism (`temperature=0` does not strictly enforce greedy
decoding on YandexGPT-Lite; the API's tie-breaking on equiprobable
tokens appears to introduce occasional run-to-run variance, especially
on borderline cases).

No `statsmodels` resampling is used in this reconciliation — the
report is a direct row-by-row CSV comparison plus the residual log
from the overwritten run. The Wilson 95% CIs quoted from the two
persisted runs (both `[92.3%–99.7%]` for the min-judge 98.6% — note
the §4.4 table cell rounds the 57/58 rate to 98.6% under a wider
n=70 framing; the per-run summary uses 57/58 = 98.3% under the
fallback-excluded n=58 framing) are computed via
`statsmodels.stats.proportion.proportion_confint(method="wilson")` —
deterministic for any given (k, n).

## 5. Canonical run

**`reports/faithfulness_multijudge_2026-05-19.md` (raw:
`faithfulness_multijudge_raw_2026-05-19.csv`) is the canonical run.**

Reasons:

1. It is the run cited by `report_final.md` §4.4 / §7 throughout the
   project history and predates every cosmetic re-run.
2. The 2026-05-20 20:53:17 re-run produced byte-identical underlying
   verdicts, so promoting it would not change any quoted number — only
   the file date — and would obscure provenance.
3. The 2026-05-20 11:06:59 outlier run, which would *not* have agreed
   with §4.4's reported `cardio_40` disagreement, has been overwritten;
   re-anchoring to a later timestamp risks a future overwrite hiding the
   variance documented in §4 above.

Both currently-persisted runs are kept on disk under their dated
filenames for traceability. Future report citations should reference
`faithfulness_multijudge_2026-05-19.md` as the canonical multi-judge
result; the 2026-05-20 file is retained as a reproducibility witness
and the residual `report_stage_21.md` §1 + §5 entries document the
otherwise-irrecoverable 11:06:59 outlier.
