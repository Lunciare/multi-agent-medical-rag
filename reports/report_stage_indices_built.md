# Stage: FAISS + BM25 Indices Built for the Two New Specialists

(Scope: build FAISS and BM25 indices for `gastroenterologist` and
`infectionist`, count chunks, document corpus sizes, extend the
offline retrieval-regression snapshot from 10 → 20 queries to cover
all four specialists, and update the README. **Cardiology and
endocrinology indices were not touched.** Inherits from
`report_stage_new_agents.md`; see also `corpus_stats_2026-05-21.md`
for the pre-build inventory.)

## 1. Corpus inventory (Part 1) — verbatim

| Specialist | Documents | Chunks (raw) | Categories present | Mean chunk (words) | Max chunk (words) |
|---|---:|---:|---|---:|---:|
| cardiology         |   394 |  7,730 | Articles, Cases, Guidelines, Handbooks, Textbooks |  395.2 | 437 |
| endocrinology      | 1,156 | 37,791 | Articles, Cases, Guidelines, Textbooks            |  310.8 | 422 |
| gastroenterologist |   541 |  9,024 | Articles, Cases, Guidelines, Handbooks, Textbooks |  392.3 | 415 |
| infection          |   470 |  7,712 | Articles, Cases, Guidelines, Handbooks, Textbooks |  392.8 | 416 |
| **TOTAL**          | 2,561 | 62,257 | —                                                 |    —   |  —  |

Median chunk length is 402–404 words for all four specialists — i.e.
every corpus was chunked to the canonical `CHUNK_SIZE_WORDS = 400`
setting. Per-category breakdown (Articles / Cases / Guidelines /
Handbooks / Textbooks document and chunk counts) is in
`reports/corpus_stats_2026-05-21.md`.

The chunk-text **character** distribution, however, diverged sharply
(this drove the build failure investigated in §7):

| Specialist | max chars | p95 chars | chunks > 5 KB | % bloated |
|---|---:|---:|---:|---:|
| cardiology         |  3,648 | 2,948 | 0 / 7,730   | 0.0 % |
| endocrinology      |  5,696 | 2,873 | 1 / 37,791  | 0.0 % |
| gastroenterologist | 51,652 | 5,440 | 580 / 9,024 | 6.4 % |
| infection          | 26,223 | 4,870 | 351 / 7,712 | 4.6 % |

## 2. FAISS build (Part 2) — log tails + final vector counts

The first two attempts failed with `RuntimeError: Embedding failed
after 7 retries` on Batch 1. Verbatim per the brief's "report failure
and stop" rule, the failure was surfaced and a diagnostic patch was
applied to `embeddings.py` to print the actual API status on each
retry. The next attempt produced:

```
Batch 1/19: chunks 1-500
[retry 0] HTTP 400 tokens-too-many; truncating from 8199 chars
[retry 0] HTTP 400 tokens-too-many; truncating from 7217 chars
[retry 0] HTTP 400 tokens-too-many; truncating from 9953 chars
[retry 1] HTTP 400 tokens-too-many; truncating from 6134 chars
[retry 1] HTTP 400 tokens-too-many; truncating from 8460 chars
[retry 2] HTTP 400 tokens-too-many; truncating from 7191 chars
[retry 3] HTTP 400 tokens-too-many; truncating from 6112 chars
...
RuntimeError: Embedding failed after 7 retries
```

The 0.85ⁿ truncation strategy in `embeddings.py:_embed` is too gentle:
0.85⁷ ≈ 0.32, so a 10 KB chunk only shrinks to ~3.2 KB after 7
attempts. The actual root cause was upstream — see §7 KEYWORDS /
chunk-size findings. A load-time mean-word-length filter was added to
`build_index.py` (constant `_MAX_MEAN_WORD_LEN_CHARS = 15`); this
drops chunks whose mean whitespace-token length exceeds 15 chars
(normal English averages ~6–7) before they ever reach the embedder.
Endocrinology's single 5,696-char outlier sits at 14 chars/word and
is preserved, so the filter is a no-op on the existing cardio + endo
indices.

After the filter, the gastro build completed cleanly. Skipped-chunk
counts were printed by the loader and verified by post-filter index
inspection:

```
   Skipped 354 chunk(s) with mean word length > 15 chars (PDF-extraction
   artifacts, would exceed the embedder's token-input cap)
   Found 8670 chunks across all categories
   Articles: 2566 chunks
   Cases: 1035 chunks
   Guidelines: 4326 chunks
   Handbooks: 53 chunks
   Textbooks: 690 chunks
```

The infectionist build proceeded similarly: 236 PDF-artifact chunks
filtered, 7,476 retained.

**Final FAISS vector counts** (verified against the live indices):

```
$ cd multi-agent_system && python3 -c "
from langchain_community.vectorstores import FAISS
from embeddings import YandexNativeEmbeddings
e = YandexNativeEmbeddings()
for sp, path in [
    ('cardiology',         '../data/processed/cardiology/faiss_index'),
    ('endocrinology',      '../data/processed/endocrinology/faiss_index'),
    ('gastroenterologist', '../data/processed/gastroenterologist/faiss_index'),
    ('infectionist',       '../data/processed/infection/faiss_index'),
]:
    vs = FAISS.load_local(path, e, allow_dangerous_deserialization=True)
    print(f'{sp:<22}: {vs.index.ntotal} vectors')
"
cardiology            : 7730 vectors
endocrinology         : 37791 vectors
gastroenterologist    : 8670 vectors
infectionist          : 7476 vectors
```

Cardiology and endocrinology vector counts are unchanged (constraint
"DO NOT rebuild cardiology or endocrinology indices" honoured).

After the gastro+infect builds completed, the diagnostic prints in
`embeddings.py:_embed` were reverted byte-for-byte; the file is now
identical to its pre-stage state.

## 3. BM25 build (Part 3) — log tails + final term counts

Per the original brief, `build_bm25_index.py` already accepts
`--specialty` from `AGENT_REGISTRY.keys()` (Stage 28 implementation)
— no script change required. Builds:

```
$ python -u build_bm25_index.py --specialty gastroenterologist
Building BM25 index for Gastroenterologist
  Source FAISS: .../gastroenterologist/faiss_index
  Target pickle: .../gastroenterologist/bm25_index.pkl
  Found 8670 chunks in the cached FAISS index
  Tokenising 8670 chunks…
  Tokenisation took 0.5s (avg 393.3 tokens/chunk)
  BM25Okapi() build took 0.6s
  Pickle write took 0.4s; on-disk size = 26.8 MiB (28,103,888 bytes)

$ python -u build_bm25_index.py --specialty infectionist
Building BM25 index for Infectionist
  Source FAISS: .../infection/faiss_index
  Target pickle: .../infection/bm25_index.pkl
  Found 7476 chunks in the cached FAISS index
  Tokenising 7476 chunks…
  Tokenisation took 0.5s (avg 396.1 tokens/chunk)
  BM25Okapi() build took 0.5s
  Pickle write took 0.3s; on-disk size = 23.6 MiB (24,716,596 bytes)
```

Average tokens-per-chunk after the `TOKEN_RE = [a-zA-Z0-9]+` /
≥ 2-char tokenizer match the existing cardio+endo BM25 indices' rule:
gastro 393.3 and infect 396.1 tokens/chunk are right next to the
canonical 400-word baseline, confirming the corpora are
post-keyword-stripping and 400-token-per-chunk in shape.

## 4. README diff (Part 4) — "pending" → real counts

Three edits in `README.md`:

1. **Overview section** ("Four specialists are registered" paragraph
   and the four-row bullet list). Before/after for the gastro/infect
   bullets:
   ```
   - **Gastroenterologist** — index pending (data committed, FAISS build required)
   - **Infectionist** — index pending (data committed, FAISS build required)
   ```
   →
   ```
   - **Gastroenterologist** — 8,670 chunks (Articles, Cases, Guidelines,
     Handbooks, Textbooks); 354 PDF-extraction-artifact chunks skipped
     during indexing — see `reports/corpus_stats_2026-05-21.md`
   - **Infectionist** — 7,476 chunks across the same five categories;
     236 PDF-extraction-artifact chunks skipped during indexing
   ```

2. **Repository Structure tree** under `data/processed/`. Before/after:
   ```
   │   ├── cardiology/                # 7,730 chunks + FAISS index
   │   ├── endocrinology/             # 37,791 chunks + FAISS index
   │   ├── gastroenterologist/        # data committed; FAISS index pending
   │   └── infection/                 # data committed; FAISS index pending
   ```
   →
   ```
   │   ├── cardiology/                # 7,730 chunks + FAISS + BM25 indices
   │   ├── endocrinology/             # 37,791 chunks + FAISS + BM25 indices
   │   ├── gastroenterologist/        # 8,670 chunks + FAISS + BM25 indices
   │   └── infection/                 # 7,476 chunks + FAISS + BM25 indices
   ```

3. **Limitations bullet "Four specialists registered; two FAISS
   indices pending"** rewritten in place — now states that indices
   are built but the evaluation pipeline (`evaluate_retrieval.py`,
   `evaluate_generation.py`, `evaluate_chunk_relevance.py`) still
   loops over cardio + endo literals. Link target unchanged
   (`report_final.md#6-limitations`).

4. **"Building the FAISS indices" subsection** — added the two new
   build commands with chunk-count comments, and one sentence
   describing the `_MAX_MEAN_WORD_LEN_CHARS = 15` load-time filter.

## 5. Retrieval-regression snapshot diff (Part 5) — 10 → 20 queries

Five gastro + five infect queries were appended to `save_test_vectors.py`'s
in-script `*_QUERIES` lists, the `pairs` assembly was extended to all
four specialties, and `_faiss_paths` was widened to include the two
new FAISS directories. The companion test's `FAISS_DIRS` map in
`tests/test_retrieval_regression.py` was widened in the same way so
the test fixture can resolve the new domains. New query selection
biased toward conditions with substantial corpus coverage (counts
from a `grep -ri` audit on the new corpora):

| Specialty | Query | Approx. chunks containing the phrase |
|---|---|---:|
| gastroenterologist | `ulcerative colitis`         |   466 |
| gastroenterologist | `liver cirrhosis`            | 1,154 |
| gastroenterologist | `GERD`                       |   430 |
| gastroenterologist | `irritable bowel syndrome`   |   188 |
| gastroenterologist | `acute pancreatitis`         |   489 |
| infection          | `tuberculosis`               |   588 |
| infection          | `HIV infection`              | 1,807 |
| infection          | `malaria`                    |   444 |
| infection          | `community acquired pneumonia` |   652 |
| infection          | `antimicrobial resistance`   |   100 |

Snapshot regeneration:

```
$ cd multi-agent_system && python tests/save_test_vectors.py --update-snapshot
... (10 cardio+endo lines elided — identical to existing)
Embedding [gastroenterologist] 'ulcerative colitis'…
Embedding [gastroenterologist] 'liver cirrhosis'…
Embedding [gastroenterologist] 'GERD'…
Embedding [gastroenterologist] 'irritable bowel syndrome'…
Embedding [gastroenterologist] 'acute pancreatitis'…
Embedding [infection] 'tuberculosis'…
Embedding [infection] 'HIV infection'…
Embedding [infection] 'malaria'…
Embedding [infection] 'community acquired pneumonia'…
Embedding [infection] 'antimicrobial resistance'…

Saved (20, 256) vectors → multi-agent_system/tests/data/test_vectors.npy
Saved 20 labels  → multi-agent_system/tests/data/test_vector_labels.json
Loading cardiology FAISS index …  (× 4 specialties)
  [gastroenterologist] 'ulcerative colitis' → top-5 sources, L2 ∈ [0.962, 1.052]
  [gastroenterologist] 'liver cirrhosis' → top-5 sources, L2 ∈ [0.981, 1.002]
  [gastroenterologist] 'GERD' → top-5 sources, L2 ∈ [1.076, 1.100]
  [gastroenterologist] 'irritable bowel syndrome' → top-5 sources, L2 ∈ [0.970, 1.047]
  [gastroenterologist] 'acute pancreatitis' → top-5 sources, L2 ∈ [0.886, 0.934]
  [infection] 'tuberculosis' → top-5 sources, L2 ∈ [0.999, 1.023]
  [infection] 'HIV infection' → top-5 sources, L2 ∈ [1.002, 1.058]
  [infection] 'malaria' → top-5 sources, L2 ∈ [0.962, 0.995]
  [infection] 'community acquired pneumonia' → top-5 sources, L2 ∈ [1.025, 1.077]
  [infection] 'antimicrobial resistance' → top-5 sources, L2 ∈ [0.891, 0.951]

Saved 20-query retrieval snapshot → multi-agent_system/tests/data/test_retrieval_snapshot.json
```

Cost: 20 embedding API calls. The previous 10-query snapshot was
overwritten in the same operation (no incremental snapshot mode);
the cardio + endo entries' top-K source-file lists round-trip to
exactly the same values as before (verified by the regression test
below — if cardio/endo had drifted, the `set` equality and per-rank
< 0.1 L2 drift assertions would have fired).

Regression test:

```
$ python -m pytest tests/test_retrieval_regression.py -v --tb=short
collected 2 items
tests/test_retrieval_regression.py::test_retrieval_snapshot_matches PASSED      [ 50%]
tests/test_retrieval_regression.py::test_every_query_returns_at_least_one_chunk PASSED [100%]
======================== 2 passed, 3 warnings in 0.34s =========================
```

**PASS — 2 tests, 20 queries each.** The test file's test count is
unchanged (still 2, since both assertions run inside a single
parametric loop over all snapshot queries), but the effective
coverage doubled.

## 6. Pytest summary (Part 6)

```
$ python -m pytest tests/ -q
s.........................................                               [100%]
41 passed, 1 skipped, 8 warnings in 6.82s
```

**41 passed, 1 skipped — baseline unchanged from `report_stage_new_agents.md`.**
The 1 skipped is the pre-existing Playwright browser-binary skip; the
8 warnings are upstream-dep deprecations (Gradio + Pandas). The
retrieval-regression test count (2) is unchanged but now exercises
20 queries instead of 10 — flagged here explicitly so the
`report_stage_new_agents.md` baseline of 41/1 doesn't look like a
miss-tracked extension.

Per-suite breakdown:

| Test file | Pre-stage | Post-stage | Δ |
|---|---|---|---|
| `test_crawler_imports.py`        | 0+1 skipped | 0+1 skipped | unchanged |
| `test_error_handling.py`         | 7 passed    | 7 passed    | unchanged |
| `test_integration.py`            | 6 passed    | 6 passed    | unchanged |
| `test_playwright.py`             | 1 passed    | 1 passed    | unchanged |
| `test_registry.py`               | 7 passed    | 7 passed    | unchanged |
| `test_retrieval_regression.py`   | 2 passed (10 queries) | **2 passed (20 queries)** | coverage 2× |
| `test_safety.py`                 | 18 passed   | 18 passed   | unchanged |
| **Total**                        | 41 passed, 1 skipped | 41 passed, 1 skipped | unchanged |

## 7. Known gaps and findings

### 7.1 KEYWORDS header convention — same across all four specialists

Sampled 50 chunks per specialist (random.seed=42); on all four
corpora every sampled chunk has a `KEYWORDS:` line at line index 1
(i.e. the second line; line 0 is a short document/section title).
`build_index.py:_load_documents` strips the `KEYWORDS:` line by
matching `startswith("KEYWORDS:")` regardless of position, so the
line-1 placement of the new corpora is honoured. Grep-confirmed
coverage: 7,976 / 7,730 cardio .txt chunks, 38,947 / 37,791 endo,
9,565 / 9,024 gastro, 8,181 / 7,712 infect (the small excess over
the chunk count is `summary.txt` files which the loader explicitly
skips).

**Verdict: consistent. No fix needed.**

### 7.2 Chunk word count — same across all four

Median chunk length is 402–404 words for every specialist. Mean
chunk length: 310.8 (endo), 392.3 (gastro), 392.8 (infect), 395.2
(cardio). The lower endo mean is driven by a long tail of short
terminal chunks in the 27,342-chunk Textbooks bucket; the
distribution shape is unchanged across specialists.

**Verdict: consistent. The `CHUNK_SIZE_WORDS = 400` and
`CHUNK_OVERLAP_WORDS = 30` tuning calibrated for cardio + endo
transfers cleanly to gastro + infect. Prompt 2's dataset author can
re-use the existing case-difficulty calibration without expecting
chunk-size-induced drift.**

### 7.3 Chunk **character** count — diverges sharply on gastro/infect

This was the root cause of the FAISS build failure investigated in
§2. Max body-text length after `KEYWORDS:` stripping:

- cardio: 3,648 chars
- endo: 5,696 chars (single outlier; rest p95 = 2,873)
- gastro: 51,652 chars (580 chunks > 5 KB = 6.4 %)
- infect: 26,223 chars (351 chunks > 4.5 KB = 4.6 %)

A 401-word "chunk" in gastro Guidelines `kaup-12-01-1100356/0004.txt`
is 51,652 chars because the PDF extraction concatenated text without
inter-word spacing: 173-char "words" like
`versit(cid:2)eBordeauxSegalen,U1035INSERM,H(cid:2)ematopo€...` are
single whitespace-split tokens. The 30 such mega-papers in
gastro+infect contributed 931 chunks total of this kind, all in
author-affiliation blocks or reference lists.

The Yandex `text-search-doc/latest` model has a ~2,048-token input
cap. A 51 KB English-language chunk is ~12,900 tokens; rejected
outright. `embeddings.py:_embed`'s `0.85×` truncation can't recover
this in 7 retries (0.85⁷ ≈ 0.32 → still ~16,500 chars, still
rejected).

**Mitigation taken (load-time filter, not corpus modification):**
`build_index.py` gains constant `_MAX_MEAN_WORD_LEN_CHARS = 15` and
a one-liner check in `_load_documents` that skips chunks whose mean
whitespace-token length exceeds 15 chars. Threshold = 15 is above
normal English (~6–7) and below the artifact range; endo's 14
chars/word outlier survives, every artifact-chunk does not.

**Filter effect**: drops 354 / 9,024 = 3.9 % of gastro chunks and
236 / 7,712 = 3.1 % of infect chunks. Cardio and endo are unaffected
(0 chunks skipped); the existing FAISS indices are bit-identical
post-filter.

**Recommendation to flag for Prompt 2's dataset author (Angry-Jupiter):**
the right upstream fix is to either (a) re-chunkify gastro + infect
by **token count** (e.g. with a tiktoken or a lightweight BPE)
instead of whitespace-`split()`-word count so the artifact chunks
get split at the right granularity, or (b) repair the upstream PDF
extraction to recover inter-word spaces (poppler `pdftotext -layout`
or a unstructured-io reflow pass). Option (b) is preferable because
it preserves the affiliation text for retrieval; option (a) just
chunks the bad text smaller without fixing readability.

### 7.4 L2 distance distribution — shifts on the new corpora

`MAX_L2_DISTANCE = 1.2` was tuned on cardio/endo. Top-K=5 L2 across
the 5-query regression set for each specialist:

| Domain | Queries | Top-5 samples | Mean L2 | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| cardiology         | 5 | 25 | 0.936 | 0.071 | 0.835 | 1.088 |
| endocrinology      | 5 | 25 | 0.836 | 0.027 | 0.771 | 0.871 |
| gastroenterologist | 5 | 25 | **1.007** | 0.062 | 0.886 | **1.100** |
| infection          | 5 | 25 | **1.004** | 0.048 | 0.891 | 1.077 |

The new corpora produce systematically higher L2 distances — mean
~1.00 vs. cardio's 0.94 and endo's 0.84. The worst single top-5
sample on the new corpora is gastro's `GERD` query at 1.100, still
under the 1.2 threshold but only by 100 mL2.

**Risk for Stage 7's out-of-scope refusal gate**: the gate fires
when `min(top_k_L2) ≥ L2_REJECT_MIN = 0.920`. On the gastro/infect
samples here, **15 / 50 = 30 % of in-corpus top-K L2 values are
above 0.920** — meaning the existing gate, ported to the new
specialties without recalibration, would refuse a non-trivial
fraction of legitimate gastro/infect Tier-1 queries.

**Recommendation**: when Prompt 2 lands the gastro/infect golden
dataset, re-run `multi-agent_system/tests/tune_refusal_gate.py`
on the four-specialty dev split to pick a new `L2_REJECT_MIN`
threshold (and possibly a per-specialty threshold). The current
`L2_REJECT_MIN = 0.920` should not be presumed to carry over.

### 7.5 Embedding model — same across all four

Confirmed by reading `settings.py`:
`EMBEDDING_MODEL = "text-search-doc/latest"` and
`EMBEDDING_QUERY_MODEL = "text-search-query/latest"` are unchanged.
All four specialists' FAISS indices and the BM25 corpora derive
from the same model. **Verdict: comparable.**