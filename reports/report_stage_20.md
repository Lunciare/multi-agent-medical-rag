# Stage 20 — Disclose Auto-Annotation Circularity in `reports/report_final.md`

## 1. What Was Changed

Two prose-only insertions into `reports/report_final.md` that explicitly
disclose the methodological circularity already identified in Stage 6 §4:
`gold_sources` are produced by retrieving from the same FAISS+embedding stack
that Recall@K is then computed against, so the metric measures the system's
ability to surface keyword-positive top-20 documents into the top-5 window,
not ground-truth retrieval against an independently labelled corpus.

No numerical values, table cells, confidence intervals, or other prose were
modified. Only the two sentences below were added.

### 1.1 Insertion in §4.3 (Retrieval Hit Rate)

Inserted immediately after the first sentence of the existing
"**Note on Recall@K denominators:**" paragraph (line 266). Verbatim text added:

> These gold sources were auto-annotated by retrieving top-20 chunks from the
> same FAISS+embedding system being evaluated, then keyword-filtered;
> Recall@K therefore measures the system's ability to surface
> keyword-positive top-20 documents into the top-5 window, not ground-truth
> retrieval against an independently labelled corpus.

Note: the user-approved adjustment from the original spec changed
"produced by retrieving the top-20 chunks" → "auto-annotated by retrieving
top-20 chunks" so that the smoke-test pattern matches in both locations
(see §3 below). Substance and meaning are unchanged.

### 1.2 Insertion in §7 (Conclusion)

Appended as the final sentence of the retrieval bullet that begins
"**Retrieval Recall@K on the held-out test split:**" (line 454). Verbatim
text added:

> Note: Recall@K denominators were auto-annotated by retrieving top-20
> chunks from the same FAISS+embedding system being evaluated; the metric
> therefore measures the system's ability to surface keyword-positive
> top-20 documents into the top-5 window, not ground-truth retrieval
> against an independently labelled corpus.

## 2. Why

Stage 6 §4 (`reports/report_stage_6.md`) already documents the bias in plain
language: "The auto-annotation introduces a known bias — gold documents are
sampled from keyword-hit chunks within the top-20 ... Recall@K therefore
measures 'how well does retrieval surface the kw-positive top-20 docs into
the top-5 window' rather than ground truth from the full corpus." Until this
stage that disclosure lived only in the per-stage report. The headline
metric appears in two places in `report_final.md` (§4.3 table and §7
Conclusion bullet); both now carry the circularity caveat inline so a reader
who only consults the final report cannot miss it.

## 3. Smoke-Test Output

Command (verbatim from the task spec):

```
grep -c "auto-annotated by retrieving top-20" reports/report_final.md
```

Output:

```
2
```

Expected: `2`. PASS.

The two matches are at:
- line 266 (§4.3 "Note on Recall@K denominators:" paragraph)
- line 454 (§7 Conclusion, retrieval Recall@K bullet)

## 4. Confirmation: No Numerical Values Changed

`git diff --unified=0 reports/report_final.md` shows exactly two changed
lines (266 and 454). Both diffs are pure prose additions — every numeric
token in the surrounding text is byte-identical to the prior version:

- §4.3 paragraph: counts `82`, `84`, `3`, `≥1`, `20`, case IDs `cardio_35`
  and `endo_46`, the Tier 3 reference, and `gold_sources: []` are all
  unchanged.
- §7 retrieval bullet: `56.2%`, `(86/153)`, `[Wilson 95% CI 48.3%–63.8%]`,
  `56.6%`, `(43/76)`, `[45.4%–67.1%]`, `55.8%`, `(43/77)`,
  `[44.7%–66.4%]`, `88.6%`, `(62/70)`, `[79.0%–94.1%]`, `59.0%`, `54.1%`,
  `100%`, `78.6%` are all unchanged.

The Recall@K, MRR@K, KeywordHitRate, Refusal Rate, and Wilson CI tables in
§4.3 and §4.6 are untouched. No headline number in this report has been
revised; only the methodological disclosure surrounding the existing numbers
has been strengthened.

## 5. Files Touched

- `reports/report_final.md` — two single-line prose insertions
- `reports/report_stage_20.md` — this stage report (new)

## 6. Commit Message

```
[docs] disclose auto-annotation circularity in §4.3 and §7
```
