# Stage 22 — External Benchmark Retrieval Evaluation Against PubMedQA

## 1. What Was Changed

Adds a first external-benchmark retrieval evaluation for the cardiologist
agent's FAISS index. The new harness loads PubMedQA's 1000-case expert
labelled subset (`qiaojin/PubMedQA`, subset `pqa_labeled`, Jin et al.
2019), filters to cardiology-relevant questions via a 12-keyword OR, and
computes a sentence-level Jaccard-based Recall@5 against the gold abstract
passages stored on each record. This is the first retrieval number on the
project that uses an **independently labelled corpus**, addressing the
auto-annotation circularity disclosed in §4.3 of the final report.

Two interpretive deviations from the original spec are documented inline
(both surfaced as questions and user-approved before landing):

1. **Matching granularity** — the spec described "sentence-level overlap"
   followed by a token-level Jaccard formula. Passage-level Jaccard on the
   full ~400-word chunks vs ~50-150-word PubMedQA passages was
   structurally capped near 0.20 even for perfect content matches, yielding
   0/275 hits. The harness was switched to true sentence-level matching:
   split both chunks and passages on `[.!?]`, compute Jaccard for every
   (chunk_sentence, gold_sentence) pair, hit if the max ≥ threshold.
2. **Threshold** — the spec called for ≥ 0.30. A probe across all 275 gold
   passages found the maximum sentence-pair Jaccard achievable was 0.294
   (mean 0.163), because the cardiology corpus and PubMedQA share domain
   vocabulary but differ in register (clinical-guideline / textbook vs
   research-abstract). The threshold was relaxed to ≥ 0.20 — the 21.5%
   percentile of the achievable distribution — which surfaces a non-zero
   comparison signal without being dominated by stopword overlap. The §4.9
   narrative documents this trade-off and explains why PubMedQA itself
   does not define a canonical Jaccard cutoff.

## 2. Filtered PubMedQA Count

- Loaded `qiaojin/PubMedQA`, subset `pqa_labeled`, split `train` — 1000
  expert-labelled QA pairs.
- Filter: case-insensitive substring OR over `{heart, cardiac, cardio,
  ventricular, atrial, coronary, mitral, aortic, valve, arrhythmia,
  hypertension, stroke}`.
- **Filtered count: 85 cardiology-relevant questions** carrying a total of
  **275 gold abstract passages** across them.

(Smoke-test verification with the spec's 9-keyword variant returned 67
cases ≥ 30 threshold; the full 12-keyword variant in the eval adds
`valve`, `hypertension`, `stroke` and yields 85.)

## 3. Per-Case Recall@5 with Wilson 95% CI

From [`reports/external_pubmedqa_2026-05-20.md`](external_pubmedqa_2026-05-20.md)
(written 2026-05-20 21:24:03):

| Metric | Value | 95% Wilson CI |
|---|---|---|
| Hits / gold passages | 59 / 275 | — |
| **Pooled Recall@5** | **21.5%** | **[17.0%–26.7%]** |

Stdout:

```
============================================================
  External (PubMedQA cardiology) Recall@5 Complete
============================================================
  Filtered cases:       85
  Gold passages:        275
  Hits (Jaccard>=0.2):  59
  Recall@5:             21.5% [Wilson 95% CI 17.0%–26.7%]
```

(The earlier stdout line read `Hits (Jaccard>=0.3)` as a hard-coded label
during the threshold-tuning iteration; fixed in
`tests/evaluate_external.py:255` to interpolate `{JACCARD_THRESHOLD}`. The
markdown report itself always used the live threshold via f-string.)

Per-passage distribution of the best sentence-pair Jaccard achieved
across the top-5 retrieved chunks (across all 275 trials):

| Threshold | Hits | Pooled Recall@5 |
|---|---|---|
| ≥ 0.30 | 0 | 0.0% |
| ≥ 0.25 | 15 | 5.5% |
| ≥ 0.20 (operating point) | 59 | 21.5% |
| ≥ 0.15 | 170 | 61.8% |
| ≥ 0.10 | 258 | 93.8% |

The 0.20 operating point sits on the steep part of the curve and is the
defensible mid-point that surfaces a non-zero signal without crossing into
stopword-driven noise.

## 4. Verbatim §4.9 Text Added to `reports/report_final.md`

```markdown
### 4.9 External Benchmark: PubMedQA Cardiology Slice

To anchor the in-house Recall@K against an independently labelled
biomedical retrieval benchmark — addressing the auto-annotation
circularity disclosed in §4.3 — we evaluate the cardiologist agent's FAISS
index against PubMedQA's expert-labelled subset \cite{jin2019pubmedqa},
downloadable from HuggingFace as `qiaojin/PubMedQA`, subset `pqa_labeled`
(1000 manually curated yes/no/maybe research-question QA pairs). Filtering
the 1000-case split to cardiology-relevant questions via a case-insensitive
substring OR over {`heart`, `cardiac`, `cardio`, `ventricular`, `atrial`,
`coronary`, `mitral`, `aortic`, `valve`, `arrhythmia`, `hypertension`,
`stroke`} yields **n=85 questions** with 275 gold abstract passages across
them.

| Source | Recall@5 (pooled) | n (gold trials) | 95% Wilson CI |
|---|---|---|---|
| This work (in-house, held-out test split, cardiology) | 56.6% (43/76) | 76 gold-doc Bernoulli trials | [45.4%–67.1%] |
| PubMedQA cardiology slice (sentence-level Jaccard ≥ 0.20) | 21.5% (59/275) | 275 gold-passage Bernoulli trials | [17.0%–26.7%] |

Matching threshold: each retrieved chunk and each gold passage is split
into sentences on `[.!?]` boundaries, tokens are lowercased alphanumeric
words of length ≥ 2, and a chunk is judged to *hit* a gold passage when at
least one (chunk_sentence, gold_sentence) pair reaches token-level Jaccard
`|A ∩ B| / |A ∪ B|` ≥ 0.20. The spec's preferred threshold (≥ 0.30) was
empirically unreachable on this corpus pair — a probe across all 275 gold
passages found the maximum achievable sentence-pair Jaccard was 0.294
(mean 0.163), because the cardiology corpus is written in clinical-guideline
/ textbook register while PubMedQA passages are research-abstract register.
0.20 sits at the 21.5% percentile of the achievable distribution and is
the operating point that surfaces a non-zero comparison signal without
being dominated by stopword overlap. PubMedQA itself (Jin et al. 2019
\cite{jin2019pubmedqa}) does not define a canonical Jaccard threshold for
retrieval matching — it uses BERT-based reading-comprehension evaluation
against a single labelled answer. The Jaccard-based matching here is a
deliberately simple lexical surrogate chosen so the per-passage hit rule
is reproducible without any judge LLM, accepting that it under-counts
semantically correct retrievals that paraphrase rather than lexically
overlap. The two rows in the table above are therefore not directly
comparable: the in-house row uses doc-level identity matching against gold
sources auto-annotated from the same retrieval system (the very
circularity disclosed in §4.3), whereas the PubMedQA row uses lexical
Jaccard matching against an independently labelled corpus from a
different register entirely. The 35-point gap is consistent with both
interpretations — (a) the in-house number is inflated by the same-FAISS-system
gold-source bias, and (b) the PubMedQA Jaccard rule under-counts
paraphrastic matches — and we cannot, on this data, separate the two
contributions. The external Recall@5 is reported here as a directional
sanity-check, not as a head-to-head comparison; the implementation lives
in `tests/evaluate_external.py` and the per-question table is in
`reports/external_pubmedqa_2026-05-20.md`.
```

## 5. Direct In-House vs PubMedQA Comparison Line

**In-house cardiology Recall@5 (held-out test split, doc-level identity
against same-system auto-annotated gold): 56.6% (43/76) [45.4%–67.1%]
vs PubMedQA cardiology slice Recall@5 (sentence-level Jaccard ≥ 0.20
against independently labelled abstracts): 21.5% (59/275) [17.0%–26.7%] —
a 35.1-point gap.**

The Wilson 95% CIs do not overlap (in-house lower bound 45.4% is 18.7
points above PubMedQA upper bound 26.7%), so the gap is statistically
robust under either metric definition. The §4.9 narrative is explicit
that the two numbers are not head-to-head comparable — the in-house
metric uses doc-level identity matching against gold sources sampled from
the same retrieval system being evaluated (the §4.3 circularity), while
the PubMedQA metric uses lexical Jaccard matching against an
independently labelled corpus from a different surface register
(clinical-guideline vs research-abstract). Both biases push in opposite
directions: the in-house number is inflated by the same-system gold-source
bias, while the PubMedQA number is depressed by the Jaccard rule
under-counting paraphrastic matches. The 35-point gap is therefore best
read as a *directional sanity-check* — the in-house numbers are
plausibly upper bounds; PubMedQA-style independent-corpus retrieval is
plausibly a lower bound — not a tight head-to-head comparison.

## 6. Files Touched

- `multi-agent_system/tests/evaluate_external.py` — new harness (216 lines)
- `multi-agent_system/requirements.txt` — added `datasets>=2.18`
- `.gitignore` — added HuggingFace cache patterns (`.cache/huggingface/`,
  `huggingface/`, `.huggingface/`, `hf_cache/`)
- `reports/report_final.md` — added §4.9 (between §4.8 and §5); added Jin et al. 2019 to §8 References
- `reports/references.bib` — added `@inproceedings{jin2019pubmedqa, ...}`
- `reports/external_pubmedqa_2026-05-20.md` — generated by the eval run (per-case table)
- `reports/report_stage_22.md` — this stage report (new)

## 7. Not Committed

Per spec, nothing is committed. Working tree holds the changes above for
the user to commit manually. The `datasets` package was installed locally
to run the smoke test and the eval; `pip install -r
multi-agent_system/requirements.txt` will pick it up for collaborators.
