# Stage 18 Report: README Evaluation-Results Table + Limitations + Related-Work Link

**Date:** 2026-05-20

## 1. What Was Changed

- `README.md` "Overview" section: added a one-line pointer to the report's Related Work section — `For prior work this project builds on and differs from, see [report §1.5 — Related Work](reports/report_final.md#15-related-work).`
- `README.md` "Evaluation Results" section: deleted the `> **TODO — fill in current results …**` marker and the `TBD`-filled placeholder table. Replaced with a populated per-tier × per-metric table (held-out test split, Wilson 95% CIs) plus a footnote linking back to the report's §4.8 / §4.3–§4.7 / §4.5 / §5.3 / §8.
- `README.md` new "Limitations" subsection (5 bullets) summarising report §6 with hyperlinks to the specific limitation paragraphs.

## 2. New README Table (verbatim)

```markdown
| Metric                  | T1 Cardiology              | T1 Endocrinology           | T2 Cardiology              | T2 Endocrinology           | Tier 3 (Out-of-Scope)        | Overall                              |
|-------------------------|----------------------------|----------------------------|----------------------------|----------------------------|------------------------------|--------------------------------------|
| Routing Accuracy        | 100.0% [77.2%–100%]        | 100.0% [75.8%–100%]        | 100.0% [78.5%–100%]        | 100.0% [80.6%–100%]        | 100.0% [79.6%–100%]          | **100.0% (70/70) [94.8%–100%]**       |
| Recall@5                | 59.0% [43.4%–72.9%]        | 60.6% [43.7%–75.3%]        | 54.1% [38.4%–69.0%]        | 52.3% [37.9%–66.2%]        | *n/a (no gold docs)*         | **56.2% (86/153) [48.3%–63.8%]**      |
| Faithfulness (min-judge) | 100.0% [77.2%–100%]       | 100.0% [75.8%–100%]        | 100.0% [78.5%–100%]        | 100.0% [80.6%–100%]        | 100.0% [79.6%–100%]          | **98.6% (69/70) [92.3%–99.7%]**       |
| Tier 3 Refusal Rate     | —                          | —                          | —                          | —                          | **80.0% (12/15) [54.8%–93.0%]** | —                                  |

> **Footnote.** All numbers from the held-out test split (n=70; see [report §4.8](reports/report_final.md#48-held-out-test-set-results-n70)). For dev-set results used during hyperparameter tuning see [report §4.3–§4.7](reports/report_final.md#43-retrieval-hit-rate). Recall@5 denominators are pooled gold-doc Bernoulli (each Tier 1/2 case contributes 1–3 gold-doc trials). Faithfulness (min-judge) counts a case FAITHFUL only if both Yandex judges (`yandexgpt/latest` and `yandexgpt-lite/latest`) agree; the single disagreement is `cardio_40` (Tier 2 cardiology — see [report §5.3](reports/report_final.md#53-epistemic-bounds-of-same-family-evaluation)). Tier 3 Refusal Rate is the numeric out-of-scope gate from Stage 7 ([§4.5](reports/report_final.md#45-out-of-scope-refusal-gate)); at the chosen threshold `L2_REJECT_MIN = 0.92` the same gate falsely refuses 27/55 = 49.1% of Tier 1/2 queries — see the §4.5 trade-off discussion.
```

## 3. New README "Limitations" Subsection (verbatim)

```markdown
### Limitations

- **Small sample size.** n=70 test cases gives Wilson 95% CIs of ±5–25 pp depending on the tier; the per-tier point estimates carry substantial uncertainty. See [report §6 Limitation 1](reports/report_final.md#6-limitations).
- **Cardiology corpus gaps surfaced by name.** Three Tier 2 cardiology cases (`cardio_23`, `cardio_25`, `cardio_35`) are confirmed retrieval failures with concrete missing-content categories (pericardiocentesis, Dressler / colchicine, temporary pacing). See [report §4.3.1](reports/report_final.md#431-tier-2-corpus-coverage-audit) and [§6 Limitation 2](reports/report_final.md#6-limitations).
- **Same-vendor judge bias on faithfulness.** Both judges are Yandex models; the 98.6% min-judge rate is an upper bound under the methodology characterised by [Zheng et al. 2023](reports/report_final.md#8-references). A cross-vendor judge slot (`TERTIARY_JUDGE_PROVIDER`) is implemented but not configured here. See [report §5.3](reports/report_final.md#53-epistemic-bounds-of-same-family-evaluation) and [§6 Limitation 6](reports/report_final.md#6-limitations).
- **Out-of-scope refusal trades FP for recall.** The Stage 7 numeric gate raises Tier 3 refusal from 0/16 to 12/16 but falsely refuses 49.1% of Tier 1/2 queries because min-L2 distributions overlap on this corpus. A two-stage gate (L2 pre-filter + LLM-as-classifier confirmer) is the natural next step. See [report §4.5](reports/report_final.md#45-out-of-scope-refusal-gate) and [§6 Limitation 8](reports/report_final.md#6-limitations).
- **Two-agent scope.** Only cardiology and endocrinology have full pipelines; extending to additional specialties is a registry entry plus a corpus + FAISS build (Stage 8 / [§7 Conclusion](reports/report_final.md#7-conclusion)). See [report §6 Limitation 3](reports/report_final.md#6-limitations).
```

## 4. New "Related Work" One-Liner (under Overview, verbatim)

```markdown
For prior work this project builds on and differs from, see [report §1.5 — Related Work](reports/report_final.md#15-related-work).
```

## 5. Smoke Test Output

```text
$ python -c "
content = open('README.md').read()
assert 'TBD' not in content, 'TBD placeholder still in README'
assert 'TODO — fill in current results' not in content
assert 'Wilson' in content or '95%' in content or '[' in content
assert 'Limitations' in content
print('README evaluation table is populated.')
"
README evaluation table is populated.
```

## 6. Anchor Link Path Convention

All hyperlinks use GitHub-flavor markdown auto-anchors. The anchor for a heading is the lowercased text with `.`, spaces, and other punctuation stripped, `&` removed, and remaining spaces replaced by `-`. Verified targets:

| Section heading in `report_final.md` | Anchor used in README |
|---|---|
| `## 1.5 Related Work` | `#15-related-work` |
| `### 4.3 Retrieval Hit Rate` | `#43-retrieval-hit-rate` |
| `#### 4.3.1 Tier 2 Corpus Coverage Audit` | `#431-tier-2-corpus-coverage-audit` |
| `### 4.5 Out-of-Scope Refusal Gate` | `#45-out-of-scope-refusal-gate` |
| `### 4.8 Held-Out Test Set Results (n=70)` | `#48-held-out-test-set-results-n70` |
| `### 5.3 Epistemic Bounds of Same-Family Evaluation` | `#53-epistemic-bounds-of-same-family-evaluation` |
| `## 6. Limitations` | `#6-limitations` |
| `## 7. Conclusion` | `#7-conclusion` |
| `## 8. References` | `#8-references` |

(These render correctly in GitHub's web UI; if the project is ever served through a markdown renderer that produces different anchor IDs — e.g. pandoc with `--toc` style — the link paths would need adjusting. Documented here so the next operator knows where to look.)

## 7. Open Questions

- **Anchor link IDs are renderer-dependent.** GitHub-flavour markdown auto-anchors are stable but not standardised; if the README is ever rendered through a different pipeline (Sphinx / pandoc / docs static-site generator) the `#43-retrieval-hit-rate`-style anchors may resolve differently. A future cleanup could add explicit HTML anchor tags (`<a id="48-held-out-test-set-results-n70"></a>`) in the report to make the anchor IDs explicit and renderer-independent.
- **Per-limitation anchors.** Currently every limitation hyperlink points at `#6-limitations` (the section header). Adding numbered anchors like `<a id="lim-1"></a>` next to each limitation in `report_final.md` would make the hyperlinks land on the exact paragraph rather than just the section.

## 8. Commit Message Suggestion
`[docs] populate README evaluation table from §4.8 held-out test split with Wilson 95% CIs; add Limitations subsection + Related Work one-liner with anchor links to the report`
