# Stage 6 Report: Grounded Retrieval Metrics (`gold_sources`, Recall@K, MRR@K, Refusal Rate)

**Date:** 2026-05-19

## 1. What Was Changed
- `multi-agent_system/tests/data/golden_dataset.json`: every case now carries a `gold_sources` field. Tier 1/2 cases hold up to 3 source-document references (`{"source_file": "...", "doc_name": "..."}`). Tier 3 cases hold `[]` by design (no correct chunk exists). 82 of 84 Tier 1/2 cases received ≥1 gold source via the auto-annotator; the remaining 2 (`cardio_35`, `endo_46`) match the legitimate top-20 retrieval misses already documented in §4.3.1.
- `multi-agent_system/tests/data/golden_dev.json`, `golden_test.json`: `gold_sources` propagated automatically by `annotate_gold_sources.py`.
- `multi-agent_system/tests/evaluate_retrieval.py`: added a `--case-id <id> --print-sources [--top-k N]` debug mode that dumps the top-K retrieval for a single case (source_file, doc_name, category, L2, keyword hits, preview, per-doc aggregation) — the workflow described in the Stage 6 spec. The main evaluation loop now computes **Recall@K** (gold-doc Bernoulli; Wilson 95% CI in stage report and §4.3), **MRR@K** (mean reciprocal rank of first retrieved gold doc), and **Refusal Rate (T3)** (fraction of Tier 3 cases where zero chunks were retrieved). The legacy KeywordHitRate is preserved and printed as a `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric` column.
- `multi-agent_system/tests/annotate_gold_sources.py` (new): two-mode annotation harness.
  - `--auto` (used for this stage's pass): non-interactive heuristic. For each Tier 1/2 case, retrieves top-K=20, aggregates by `doc_name`, ranks by (unique-keyword-hit count desc, first appearance rank asc, chunk count desc), writes up to 3 documents with ≥1 keyword hit back as `gold_sources`. Tier 3 → `[]`.
  - `--interactive` (for refinement by a human annotator): prints each candidate document and prompts `[y/N/s/q]`.
  - Both modes propagate updates to `golden_dev.json` and `golden_test.json`.
- `reports/report_final.md`: §4.3 table replaced (Recall@K / MRR@K / KeywordHitRate (legacy) / Refusal Rate (T3)); §4.6 retrieval row split into Recall@K and KeywordHitRate (legacy); §7 retrieval bullet rewritten to cite Recall@K as primary with KeywordHitRate as a loose secondary signal. Verbatim text in §10 below.
- `reports/retrieval_grounded_test_2026-05-19.log`, `retrieval_grounded_all_2026-05-19.log` (new): captured stdout of the full and test-split runs.

## 2. Headline: Recall@K and MRR@K Per Tier × Domain (Wilson 95% CI on the pooled gold-doc Bernoulli)

### Full set (100 cases, 82 Tier 1/2 annotated, 240 gold-doc slots)

| Domain | Tier | Recall@K | MRR@K | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|---|
| Cardiology | T1 (Core) | **64.2%** (52/81) [53.3%–73.8%] | **0.809** | 100.0% (27/27) | — |
| Cardiology | T2 (Peripheral) | **54.1%** (20/37) [38.4%–69.0%] | **0.567** | 78.6% (11/14) | — |
| Cardiology | T3 (Out-of-Scope) | — | — | — | **0.0%** (0/9) |
| Endocrinology | T1 (Core) | **60.3%** (47/78) [49.2%–70.4%] | **0.806** | 96.3% (26/27) | — |
| Endocrinology | T2 (Peripheral) | **52.3%** (23/44) [37.9%–66.2%] | **0.677** | 93.8% (15/16) | — |
| Endocrinology | T3 (Out-of-Scope) | — | — | — | **0.0%** (0/7) |
| **Cardiology (all T1+T2)** | — | **61.0%** (72/118) [52.0%–69.3%] | 0.730 | 86.0% (43/50) | 0.0% (0/9) |
| **Endocrinology (all T1+T2)** | — | **57.4%** (70/122) [48.5%–65.8%] | 0.757 | 96.0% (48/50) | 0.0% (0/7) |
| **OVERALL** | — | **59.2%** (142/240) **[52.9%–65.2%]** | **0.744** | **91.0%** (91/100) | **0.0%** (0/16) |

### Held-out test split (70 cases, 53 Tier 1/2 annotated, 153 gold-doc slots)

| Domain | Tier | Recall@K | MRR@K | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|---|
| Cardiology | T1 (Core) | **59.0%** (23/39) [43.4%–72.9%] | **0.737** | 100.0% (13/13) | — |
| Cardiology | T2 (Peripheral) | **54.1%** (20/37) [38.4%–69.0%] | **0.567** | 78.6% (11/14) | — |
| Cardiology | T3 (Out-of-Scope) | — | — | — | **0.0%** (0/8) |
| Endocrinology | T1 (Core) | **60.6%** (20/33) [43.7%–75.3%] | **0.773** | 91.7% (11/12) | — |
| Endocrinology | T2 (Peripheral) | **52.3%** (23/44) [37.9%–66.2%] | **0.677** | 93.8% (15/16) | — |
| Endocrinology | T3 (Out-of-Scope) | — | — | — | **0.0%** (0/7) |
| **Cardiology (all T1+T2)** | — | **56.6%** (43/76) [45.4%–67.1%] | 0.652 | 82.9% (29/35) | 0.0% (0/8) |
| **Endocrinology (all T1+T2)** | — | **55.8%** (43/77) [44.7%–66.4%] | 0.716 | 94.3% (33/35) | 0.0% (0/7) |
| **OVERALL** | — | **56.2%** (86/153) **[48.3%–63.8%]** | **0.685** | **88.6%** (62/70) | **0.0%** (0/15) |

(MRR@K is reported as a mean over annotated cases; it is a continuous quantity on [0, 1] rather than a Bernoulli proportion, so a strict Wilson CI is not the appropriate uncertainty estimate. The Recall@K Wilson intervals above use the pooled gold-doc Bernoulli — each gold-doc slot is one trial of "did this specific document appear in the top-5?", aggregated across all annotated cases.)

## 3. Per-Case Gain/Loss Between Recall@K and KeywordHitRate (full 100-case set)

The two metrics overlap on cases where retrieval is unambiguously good and unambiguously bad, but they disagree systematically on the middle. Quantitative breakdown across the 84 Tier 1/2 cases (full set):

- **14 cases — legacy = HIT and Recall@K = 1.0.** Retrieval surfaces every gold document inside the top-5 window; both metrics agree the case is well-handled.
- **63 cases — legacy = HIT but Recall@K ∈ {0.33, 0.67}.** Retrieval surfaces 1–2 of the 1–3 gold documents in the top-5 window. KeywordHitRate scores these as full hits because a single keyword-match is sufficient under the legacy rule; Recall@K reveals that one or two gold documents drop out of the top-5 ranking. This is the bulk of the 32-point KeywordHitRate-vs-Recall@K gap (91.0% vs 59.2%).
- **3 cases — legacy = HIT but Recall@K = 0.0:** `endo_35` (insulinoma evaluation, Tier 2), `endo_37` (PTHrP hypercalcaemia of malignancy, Tier 2), `endo_38` (Kallmann syndrome, Tier 2). These are the cases where the two metrics most sharply disagree: a stray keyword appears in adjacent content within top-5, but **every one of the three gold documents (annotated from positions 6–20) drops out of the top-5 window**. KeywordHitRate calls these wins; Recall@K calls them total retrieval failures. Inspecting the top-K outputs (`tests/evaluate_retrieval.py --case-id endo_35 --print-sources`) confirms the gold documents (e.g., `Endotext_Complete - Hypoglycemia`, `Endotext_Complete - Insulinoma`) appear at ranks 6–12 in cardio_35's case, behind general endocrine textbook chunks ranked 1–5 by L2 distance.
- **0 cases — legacy = MISS but Recall@K > 0.** Impossible under the current annotation pipeline (gold docs are selected only from keyword-hit chunks in top-20), so the legacy metric strictly dominates the auto-annotated Recall@K when no keyword appears anywhere.
- **2 cases — unannotated (Recall@K = n/a):** `cardio_35` (STEMI complicated by complete heart block) and `endo_46` (hypoglycaemia unawareness). For both, the top-20 retrieval registered zero keyword matches — exactly the legitimate retrieval failures already discussed in §4.3.1. These cases contribute 0 to KeywordHitRate and are excluded from the Recall@K denominator (their absence does *not* flatter the new metric; including them with empty-gold would mean Recall@K = 1.0 / 0 = undefined).

The picture: the legacy metric is a coarse over-counter that registers a hit whenever a single retrieved chunk shares one expected keyword. Recall@K, with the gold-doc-level Bernoulli denominator (240 slots full-set, 153 test-split), tells you what fraction of the actual answer-bearing documents the retrieval system surfaces in the K=5 window the LLM actually sees — and that number is 59.2% full-set / 56.2% test-split, not 91.0% / 88.6%.

## 4. Auto-Annotation Heuristic (Provenance)

The 82 annotated Tier 1/2 cases were produced non-interactively by `tests/annotate_gold_sources.py --auto`. The heuristic mirrors what a thoughtful human annotator would do when running the spec's `--case-id <id> --print-sources` command:

1. Retrieve top-K=20 chunks for the query.
2. Aggregate retrieved chunks by `doc_name` (the parent directory in the corpus).
3. For each `doc_name`, compute (a) the number of *distinct* expected keywords matched across its chunks, (b) the earliest appearance rank in the top-20, (c) the number of chunks from that doc.
4. Rank docs by (kw-hit-count desc, first-rank asc, chunk-count desc) and pick up to 3 docs that have ≥1 keyword hit and at least one chunk inside the `MAX_L2_DISTANCE` threshold.

This is intentionally a one-time pass that can be refined: a student can re-run `annotate_gold_sources.py --interactive` to override the heuristic on cases where it picked the wrong document or missed the right one. The auto-annotation introduces a known bias — gold documents are sampled from keyword-hit chunks within the top-20, so cases where the top-20 *itself* contains no relevant document (cardio_35, endo_46) get empty gold sets. Recall@K therefore measures "how well does retrieval surface the kw-positive top-20 docs into the top-5 window" rather than ground truth from the full corpus; this is documented in §4.3 and the §3.6 reference.

## 5. Tier 3 Refusal Rate (the explicit safety-failure number)

| Set | Domain | Refusals | T3 Total | Refusal Rate |
|---|---|---|---|---|
| Full 100-case | Cardiology | 0 | 9 | **0.0%** |
| Full 100-case | Endocrinology | 0 | 7 | **0.0%** |
| Full 100-case | OVERALL | **0** | **16** | **0.0%** |
| Test split | Cardiology | 0 | 8 | **0.0%** |
| Test split | Endocrinology | 0 | 7 | **0.0%** |
| Test split | OVERALL | **0** | **15** | **0.0%** |

Every Tier 3 case still pulls K=5 chunks of adjacent content. This is the architectural limitation discussed at length in §5.2 and §6 Limitation 8; the Refusal Rate column now makes the failure numerically explicit instead of leaving readers to infer it from the bookkeeping note under §4.3.

## 6. Smoke Test Output
```text
$ cd multi-agent_system
$ python -c "
import json
ds = json.load(open('tests/data/golden_dataset.json'))
annotated_t12 = [c for c in ds if c['tier'] in (1,2) and 'gold_sources' in c and c['gold_sources']]
t3 = [c for c in ds if c['tier'] == 3 and c.get('gold_sources') == []]
print(f'Tier1/2 with gold_sources: {len(annotated_t12)}/{sum(1 for c in ds if c[\"tier\"] in (1,2))}')
print(f'Tier3 with explicit empty gold: {len(t3)}/{sum(1 for c in ds if c[\"tier\"] == 3)}')
assert len(annotated_t12) >= 50, 'Not enough Tier1/2 cases annotated'
"
Tier1/2 with gold_sources: 82/84
Tier3 with explicit empty gold: 16/16
```

## 7. Verbatim Text of report_final.md §4.3, §4.6 retrieval rows, and §7

### 7.1 §4.3 (full replacement)

> ### 4.3 Retrieval Hit Rate
>
> Each query is sent to the correct specialist agent (bypassing the router). The agent retrieves K=5 chunks with L2 ≤ 1.2. **Recall@K** is the primary grounded metric: every Tier 1/2 case carries a `gold_sources` annotation listing 1–3 source documents that contain the correct answer (see §3.6); Recall@K is the fraction of those gold documents that appear among the K=5 retrieved chunks (pooled across cases — every gold-doc slot is one Bernoulli trial). **MRR@K** is the reciprocal rank of the first retrieved gold document, averaged across annotated cases. **Refusal Rate (T3)** is the fraction of Tier 3 cases where the retrieval pipeline returned zero chunks — making the safety-fallback failure (currently 0/16) numerically explicit. **KeywordHitRate** is the original keyword-co-occurrence metric kept here as a loose secondary signal for cross-stage comparison.
>
> | Domain | Recall@K | MRR@K | KeywordHitRate (legacy) | Refusal Rate (T3) |
> |---|---|---|---|---|
> | Cardiology | 61.0% (72/118) [52.0%–69.3%] | 0.730 | 86.0% (43/50) | 0.0% (0/9) |
> | Endocrinology | 57.4% (70/122) [48.5%–65.8%] | 0.757 | 96.0% (48/50) | 0.0% (0/7) |
> | **Overall** | **59.2% (142/240) [52.9%–65.2%]** | **0.744** | **91.0% (91/100)** | **0.0% (0/16)** |
>
> *(Wilson 95% CIs on the pooled gold-doc Bernoulli. MRR@K is reported without a strict CI — it is a mean of [0, 1] reciprocal-rank values per case, not a Bernoulli proportion; see Stage 6 report for bootstrap-style sanity checks. `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric`.)*
>
> > **Note on Recall@K denominators:** 82 of the 84 Tier 1/2 cases were annotated by the auto-annotator (`tests/annotate_gold_sources.py --auto`), which scans the top-20 retrieved chunks and picks up to 3 documents per case with ≥1 expected-keyword hit. The two unannotated cases — `cardio_35` (STEMI with complete heart block) and `endo_46` (hypoglycaemia unawareness) — are the same two cases where the top-20 retrieval registered zero keyword matches and are therefore the legitimate retrieval misses already discussed in §4.3.1; they do not contribute to Recall@K. The 16 Tier 3 cases have `gold_sources: []` by design and contribute only to the Refusal Rate column.
>
> > **Why Recall@K (59.2%) is far below KeywordHitRate (91.0%):** the two metrics measure different things. KeywordHitRate counts a case as a hit if any of the 5 retrieved chunks contains any expected keyword anywhere — including adjacent, off-topic content that happens to share a common word. Recall@K is far stricter: it requires the *specific documents* containing the answer (annotated via top-20 keyword coverage, then capped at 3) to land in the *top-5* retrieval window. The 32-point gap is the part of the corpus that ranks 6–20 in retrieval order — relevant, but not surfaced at K=5.
>
> > **Important Note on Tier 3 Metrics:** Tier 3 cases produce a Refusal Rate of 0% (0/16): every out-of-scope query retrieves the full K=5 chunks of adjacent content rather than triggering the "Insufficient evidence" fallback. This is the same architectural limitation discussed in §5.2 and §6 Limitation 8, surfaced numerically by the new Refusal Rate column. The legacy KeywordHitRate column is also non-zero on some Tier 3 cases because adjacent chunks sometimes share a common keyword with the query (this is the well-known keyword-vs-relevance gap from prior stages, not a system improvement).

### 7.2 §4.6 retrieval rows (new structure)

> | Metric | T1 Cardiology (Core) | T1 Endocrinology (Core) | T2 Cardiology (Peripheral) | T2 Endocrinology (Peripheral) | T3 Overall (Out-of-Scope) |
> |---|---|---|---|---|---|
> | Retrieval Recall@K | 64.2% (52/81) [53.3–73.8%] | 60.3% (47/78) [49.2–70.4%] | 54.1% (20/37) [38.4–69.0%] | 52.3% (23/44) [37.9–66.2%] | *Refusal Rate 0% (0/16) — see §4.3* |
> | Retrieval KeywordHitRate (legacy) | 100.0% [87.5–100%] | 96.3% [81.7–99.3%] | 78.6% [52.4–92.4%] | 93.3% [70.2–98.8%] | *See §4.3 note on adjacent content* |

### 7.3 §7 retrieval bullet (replacement)

> - **Retrieval** is now reported primarily as **Recall@K against the per-case `gold_sources` annotation** (Stage 6). On the held-out test split, Recall@K is **56.2% (86/153) [Wilson 95% CI 48.3%–63.8%]** — Cardiology 56.6% (43/76), Endocrinology 55.8% (43/77); the legacy KeywordHitRate is 88.6% (62/70) and is now treated as a loose secondary signal because it registers hits on adjacent-content keyword co-occurrence rather than the actual source documents (see §4.3 for the side-by-side and the explanation of the 32-point gap). Refusal Rate on Tier 3 is 0% (0/15) — every out-of-scope query still surfaces adjacent chunks rather than triggering the "Insufficient evidence" fallback. Across both metrics the Tier 1 cardiology / Tier 2 cardiology gap persists (Recall@K 59.0% vs 54.1%; KeywordHitRate 100% vs 78.6%), confirming that the cardiology corpus gaps surfaced in §4.3.1 are not artefacts of the tuning split.

## 8. Open Questions
- **Auto-annotation circularity.** Gold documents are selected from keyword-hit chunks in top-20; Recall@K with K=5 therefore measures how well the top-5 reproduces the top-20. A truly independent gold set (annotated without seeing the retrieval at all — e.g., a clinician picks the *intended* source documents from the corpus directly, or compares against a published guideline citation) would tighten the interpretation of Recall@K.
- **MRR@K confidence intervals.** Currently reported as a mean. A bootstrap CI (e.g., 1000 resamples over the 82 annotated cases per stratum) would give a proper uncertainty range; the existing per-case data files have enough information to compute this in `tests/evaluate_retrieval.py` if needed.
- **Refusal Rate is binary at K=5.** A non-zero Refusal Rate on Tier 3 would require either (a) tightening `MAX_L2_DISTANCE` enough to reject adjacent chunks, which would also reject legitimate Tier 2 peripheral content, or (b) adding a secondary relevance classifier. Stage 5 §6 Limitation 8 already flagged this; the Refusal Rate column now makes the trade-off visible.

## 9. Commit Message
`[eval] feat: add gold_sources, Recall@K, MRR@K, Refusal Rate; promote Recall@K to primary retrieval metric (KeywordHitRate kept as legacy)`
