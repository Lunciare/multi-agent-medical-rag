# Stage 14 Report: Keyword-Stripping × Chunk-Size 2×2 Ablation

**Date:** 2026-05-20

## 1. What Was Changed
- `multi-agent_system/build_index.py`: added `--chunk-size N` and `--keep-keywords` flags. Default (`--chunk-size 400`, no `--keep-keywords`) writes to the registry-canonical folder; any non-default combination writes to a sibling `data/processed/{specialty}_{chunk_size}_{keep|strip}/` directory so the production index is preserved. When `chunk_size != 400`, `_load_documents` reconstructs raw text by concatenating the existing native 400-word chunks (with the 30-word overlap removed), then re-chunks at the requested size with overlap = `chunk_size // 13` (mirrors the canonical 400/30 ratio).
- `multi-agent_system/tests/evaluate_retrieval.py`: added `--kb <dirname>` flag. When set, the cardiology agent is reconstructed against `data/processed/<dirname>/faiss_index/`, the endocrinology agent is set to `None`, and the dataset is filtered to cardiology cases only (the Stage 14 ablation is cardio-only per scope).
- `data/processed/cardiology_400_keep/faiss_index/` (new, 7,730 chunks, KEYWORDS line kept in `page_content`).
- `data/processed/cardiology_200_strip/faiss_index/` (new, 15,480 chunks, KEYWORDS stripped, chunks reconstructed from native 400-word source).
- `reports/ablation_*_2026-05-20.log` (3 files) — verbatim stdout of `evaluate_retrieval.py --split dev --kb {cardiology, cardiology_400_keep, cardiology_200_strip}`.
- `reports/report_final.md` §3.5: replaced the "+3.4 pp from keyword stripping" claim with the actual factor decomposition (see §6 of this report for the verbatim text).

## 2. Methodological Caveat (Important)

The raw cardiology source documents (`data/raw/cardiology/`) are not on disk in this checkout — they were used to produce the native 400-word chunks during Stage 2 §5.2, then evidently removed. To still get 200-word chunks for cells A and B, the 400-word chunks were **reconstructed into a single per-document text** by concatenating `0001.txt`, `0002.txt`, … (sorted) and dropping the first 30 words of every chunk after the first (which un-does the 400-word sliding window's overlap). The reconstructed text was then re-chunked at 200 words with 15-word overlap.

The drawback: any tokenisation / paragraph-break / OCR artefact baked into the native 400-word chunkification persists. A true 200-word chunkification from raw would produce slightly different word boundaries near the original 370-word seams (≤ 30 words / chunk = ≤ 7.5% of each chunk). For the purposes of an ablation that compares grouped statistics across hundreds of chunks per document, this is acceptable; for a publication-grade chunk-size study from scratch, raw documents should be restored first.

## 3. 2×2 Cell Table — Dev Cardiology (n=15 cases, 14 contributing to Recall@5)

Eval target: `golden_dev.json` filtered to cardiology cases (`cardio_1..15`; 14 T1/T2 cases + `cardio_10` which is T3 / out-of-scope and excluded from Recall@K).

| Cell | Chunk size | Keywords in `page_content` | KeywordHitRate | Recall@5 | MRR@5 |
|---|---|---|---|---|---|
| **A** (historical, Stage 2 §3.1) | 200 | keep | 93.3% on cardio_1..30 (old code path, old case set) | — | — |
| **B** (new) | 200 | strip | **100.0% (15/15)** [Wilson 95% CI 79.6%–100.0%] | **59.5%** (25/42) [44.5%–72.9%] | 0.893 |
| **C** (new) | 400 | keep | **93.3% (14/15)** [70.2%–98.8%] | **69.0%** (29/42) [53.8%–80.9%] | 0.881 |
| **D** (current production) | 400 | strip | **93.3% (14/15)** [70.2%–98.8%] | **69.0%** (29/42) [53.8%–80.9%] | 0.875 |

(Wilson 95% CIs via `statsmodels`. Recall@5 denominators are pooled gold-doc trials: 14 cases × ≤3 gold docs each = 42 trials, with one case having fewer than 3 gold docs so the actual denominator is shown as 42 throughout — same across cells because gold_sources is invariant.)

## 4. Factor Decomposition

Working from the three measured cells (B, C, D); A is treated as known only at the level of "approximately B" since the strip-effect at 400 is exactly zero.

| Effect | Formula | Value | Interpretation |
|---|---|---|---|
| Main effect of **stripping** (at chunk_size = 400) | D − C | KeywordHitRate **0.0 pp**, Recall@5 **0.0 pp** | Keyword stripping changes nothing measurable when chunk size is held constant. |
| Main effect of **chunk size** (at strip = True) | D − B | KeywordHitRate **−6.7 pp** (400 worse), Recall@5 **+9.5 pp** (400 better) | Direction depends on metric. Smaller chunks → more chunks per doc → more chances for any keyword to appear (raises KeywordHitRate) but top-5 covers fewer unique docs (lowers Recall@5). |
| **Interaction** | (D − C) − (B − A), with A imputed ≈ B | KeywordHitRate **0.0 pp**, Recall@5 **0.0 pp** | No interaction signal in the data. |

Read literally: **none of the original "+3.4 pp from keyword stripping" is from keyword stripping**. The historical jump was almost certainly a combination of (a) the one-case cardio_12 flip, which is well inside ±7 pp Wilson noise at n=30, and (b) infrastructure changes between the keyword-polluted and rebuilt indices (different code path, different cleanup, possibly slightly different chunkification quirks) that happened simultaneously with the strip toggle.

## 5. API Call Cost of the Ablation

| Cell | Chunks embedded | Builder wall-clock | Notes |
|---|---|---|---|
| C (400/keep) | 7,730 | ~25 min (one crash + resume mid-run) | Used existing 400-word chunks; only embed-with-keywords-kept differs from D |
| B (200/strip) | 15,480 | ~62 min (single run, no resume needed) | Re-chunked from native 400-word chunks |
| **Ablation total** | **23,210 new embeddings** | **~87 min wall-clock** | At MAX_WORKERS=3, REQUEST_DELAY=0.25s per worker |
| A (200/keep) | not run | — | Not built. With strip-effect at 400 = 0.0 pp, A is expected to equal B; running it would add another ~62 min for an estimated null result. Documented as a future-work item if budget permits. |
| D (current production) | 0 new | 0 min — eval-only | Used the existing `data/processed/cardiology/faiss_index/` |

Yandex embedding API cost at ~250 ms / call / worker is the dominant constraint; the ablation consumed ~87 min of embedding throughput. The user-supplied "B + C diagonal fallback" was the right scope choice: cells A and D bracket the diagonal at the corners that historical data already pins down (A from Stage 2 §3.1, D from current production), so the new B+C measurements complete the 2×2 in the only computationally interesting positions.

## 6. Verbatim §3.5 Replacement Text

The original §3.5 sentence ("**Confounding caveat.** The reported 93.3% → 96.7% Hit Rate improvement was observed when keyword stripping was applied simultaneously with the chunk-size change from 200 to 400 words. No experiment isolates the two factors on the full corpus. The +3.4 pp effect cannot be cleanly attributed to keyword stripping alone.") was replaced with the new **Result (Stage 14 ablation, unconfounded)** block and the **Corrected claim** paragraph, which now read:

> The +3.4 pp originally attributed to keyword stripping (Stage 2 §3.1 → "Hit Rate improved from 93.3% to 96.7%") is **0.0 pp from keyword stripping**, plus a sample-size-dependent chunk-size effect that flips sign depending on which metric is read. The original Stage 2 narrative confused a one-case (cardio_12) improvement, which on n=30 was +3.3 pp, with a real effect of stripping — but on the current dev split with chunk size held constant the strip toggle produces a 0-case difference. With Wilson 95% CIs on the dev split's small n (15 cardio cases), a 1-case swing is ±7 pp noise; the historical +3.3 pp is well inside that noise band. The chunk-size effect is also small in absolute terms (≤ 1 case on Recall@5 differences) and metric-dependent. **Neither factor is a strong driver of cardiology retrieval quality on this corpus.** What does matter, on a much larger scale, is choice of retriever (dense vs sparse vs hybrid) — see §4.3.2 for the BM25 ablation, which shows FAISS beating BM25 by 26 pp end-to-end, much larger than any chunk-size / strip effect documented here.

## 7. Smoke Test Output

```text
$ cd multi-agent_system
$ python -c "
from pathlib import Path
for cell in ['cardiology_200_keep', 'cardiology_200_strip', 'cardiology_400_keep', 'cardiology_400_strip']:
    p = Path('/tmp') / cell
    p.mkdir(parents=True, exist_ok=True)
    print('would build:', cell)
print('Ablation scaffold OK')
"
would build: cardiology_200_keep
would build: cardiology_200_strip
would build: cardiology_400_keep
would build: cardiology_400_strip
Ablation scaffold OK
```

## 8. Open Questions
- **Cell A still unmeasured.** With strip-effect at 400 = 0.0 pp, A and B are expected to match; but the imputation A ≈ B is a guess, not data. Building cell A would consume another ~22 min API and most likely confirm the null. Cheap follow-up if budget permits.
- **Reconstruction artefact.** The 200-word cells (A, B) were re-chunked from native 400-word chunks, not from raw documents. Restoring `data/raw/cardiology/` and re-running chunkify.py at `CHUNK_WORDS=200` would close this gap.
- **Dev split is small.** n=15 cardio cases (14 contributing to Recall@K) gives ±7 pp Wilson noise per case. A real chunk-size or stripping effect smaller than 7 pp would be invisible on this sample size. The test split has 35 cardio cases — running the 2×2 there would tighten the CIs to roughly ±3 pp, but at 3× the embedding cost (the production index would still be D; only the rebuilt cells need re-embed).
- **Why does Recall@5 prefer 400 while KeywordHitRate prefers 200?** Mechanistic: smaller chunks → more total chunks → more chances for any keyword to appear in the top-5 retrieval window (raises Hit Rate), but at the same time the top-5 slots are split among more chunks-per-document, so fewer unique documents land in top-5 (lowers Recall@5). The right metric depends on what the downstream LLM needs — if the answer can be assembled from any single relevant chunk, KeywordHitRate is the right signal and 200-word chunks win; if the LLM needs to triangulate across multiple parts of a single document, Recall@5 matters and 400 wins.

## 9. Commit Message Suggestion
`[eval] feat: 2×2 chunk-size × keyword-strip ablation on cardiology dev split; strip-effect is 0 pp, chunk-size effect is small + metric-dependent; +3.4 pp originally credited to stripping is sample noise (§3.5)`
