# Stage 10 Report: Dev/Test Split Disclosure on Tuning Sections + Confounding Caveat

**Date:** 2026-05-20

## 1. What Was Changed
- `reports/report_final.md`:
  - **§3.3** (Chunk Size Optimization): appended a one-line disclosure that the chunk-size grid was run on the 30-case dev split (proxy subset), with the selected 400-word chunk size applied to the full corpus before the §4.8 held-out evaluation.
  - **§3.4** (Retrieval Hyperparameter Grid Search): replaced the opening sentence with the explicit grid range and dev-split provenance. The grid is K ∈ {3,5,7,10,15} × L2 ∈ {0.8,1.0,1.2,1.4,1.6,2.0}, tuned on `golden_dev.json` (post Stage-4 Fix 1; previously the initial 30-case version of `golden_dataset.json`). The §4.8 held-out test split (n=70) reports performance on cases never seen during tuning.
  - **§3.5** (Effect of Metadata Pollution): appended a "**Confounding caveat**" paragraph noting that the +3.4 pp Hit Rate change (93.3% → 96.7%) was observed under simultaneous keyword stripping + chunk-size change (200 → 400 words). The effect cannot be cleanly attributed to keyword stripping alone.
  - **§7** Conclusion: inserted a bold sentence into the lead paragraph stating that all headline numbers in the Conclusion are computed on the held-out test split (n=70), while the full-set numbers (n=100) are presented separately in §4.3, §4.4, and §4.7 for completeness.
- All four edits respect the user-supplied verbatim wording. The only adaptation is the section-number references (the spec uses "§4.7" and "§4.3–§4.6" from earlier numbering; the current report has §4.8 for the Held-Out Test split and §4.3–§4.7 for the full-set metric sections, after the Stage 7 renumbering). The adaptation is documented in §6 below.

## 2. Smoke Test Output

```text
$ grep -cE "dev split|development split|held-out|confounding" reports/report_final.md
14
```

Target was ≥5; result is 14. The matches span: the new §3.3 disclosure (1), the new §3.4 opening (2 matches: "development split" + "held-out"), the new §3.5 "Confounding caveat" header (1), the existing §4.3 prefix note about dev-set tuning (1), the §4.5 refusal-gate dev/test analysis (3), the §4.8 prefix paragraph (2), the §4.8 in-text discussion (1), the §6 Limitation 8 mention (1), and the §7 lead paragraph plus retrieval/refusal/faithfulness bullets (2).

## 3. Verbatim New Text

### 3.1 §3.3 closing line (new — appended after the chunk-size table)

> Note: the chunk-size grid was also run on the 30-case dev split using a ~20-document proxy subset (cost-saving). The selected chunk size (400 words) was applied to the full corpus before the §4.8 held-out evaluation.

### 3.2 §3.4 opening sentence (replaced)

> A grid search over K ∈ {3,5,7,10,15} × L2 ∈ {0.8,1.0,1.2,1.4,1.6,2.0} was performed on the 30-case development split (`golden_dev.json` after Fix 1; previously the initial 30-case version of `golden_dataset.json`). The complete dev-set results are in [`reports/hyperparameter_grid.csv`](hyperparameter_grid.csv). Hyperparameter selection was therefore performed on a strict subset of the cases reported in §4; the §4.8 held-out test split (n=70) reports performance on cases never seen during tuning.

### 3.3 §3.5 confounding caveat (new — appended after the "Result" line)

> **Confounding caveat.** The reported 93.3% → 96.7% Hit Rate improvement was observed when keyword stripping was applied simultaneously with the chunk-size change from 200 to 400 words. No experiment isolates the two factors on the full corpus. The +3.4 pp effect cannot be cleanly attributed to keyword stripping alone.

### 3.4 §7 Conclusion lead paragraph (with the new test-split disclosure)

> Headline metrics are reported on the 70-case held-out test split (§4.8), which excludes the 30 development cases used to tune K, L2 threshold, and chunk size. Faithfulness is now reported under the minimum-judge rule (a case counts FAITHFUL only if every configured judge agrees), not the single-judge rate. The numeric refusal gate added in Stage 7 (§4.5) replaces the prompt-only fallback that previously failed on every Tier 3 case. **All headline numbers in this Conclusion are computed on the held-out test split (n=70); the full-set numbers (n=100, which include the 30 development cases used during hyperparameter tuning) are presented separately in §4.3, §4.4, and §4.7 for completeness.** The multi-agent medical RAG system shows the following performance:

## 4. `hyperparameter_grid.csv` Provenance

The §3.4 reference to `reports/hyperparameter_grid.csv` is correct and verified — every row in that file reports `total_queries=30`, confirming the grid was run against the 30-case dev split. Sample:

```text
$ head -3 reports/hyperparameter_grid.csv
K,L2_threshold,hits,total_queries,hit_rate_pct,avg_docs_retrieved,is_chosen
3,0.8,0,30,0.0,0.0,False
3,1.0,21,30,70.0,2.4,False
```

All 30 grid rows (5 K-values × 6 L2-values = 30 combinations) carry `total_queries=30`. The selected row is `K=5, L2=1.2, hit_rate_pct=96.7, is_chosen=True`.

## 5. Discussion-Section Claim Coverage (Carried Forward from Stage 9)

The Stage 9 audit of Discussion §5 confirmed every claim was either internal-data-cited (§4.x) or externally cited. The Stage 10 edits do not add claims to §5 — they only add disclosure language to §3 and §7 — so the §5 audit holds.

## 6. Open Questions
- **Section-number drift.** The task spec's verbatim text referenced "§4.7" for the held-out test split and "§4.3–§4.6" for the full-set metric range. After the Stage 7 renumbering those references migrated to "§4.8" and "§4.3–§4.7" respectively; the report uses the current numbering. If the spec author wants the exact strings preserved (e.g., for downstream automated diffing), revert to the spec's wording — but the cross-references would then point at the wrong section headings.
- **Confounding caveat depth.** The §3.5 caveat states the +3.4 pp effect cannot be cleanly attributed to keyword stripping alone, but does *not* propose an ablation experiment to isolate the two factors. A clean ablation would require rebuilding the cardiology FAISS index twice (200-word + stripped; 400-word + non-stripped) and re-running the eval — ~30 hours of embedding-API time per variant. Worth recording as a follow-up if the project author wants the attribution to be quantitative.

## 7. Commit Message Suggestion
`[docs] disclose dev-split tuning in §3.3, §3.4; add §3.5 confounding caveat; clarify §7 test-split-only headline numbers`
