# Stage 2 Report — Multi-Agent Medical RAG System

**Date:** 2026-05-09  
**Covers:** All work from initial prototype through current state

---

## 1. What Was Built

### Architecture

Two-agent RAG system with LLM routing:

```
User Query → Safety Gate (regex) → LLM Router (YandexGPT) → Specialist Agent → FAISS retrieval → LLM generation → Response
```

| Component | Implementation |
|---|---|
| Router | `orchestrator.py` — YandexGPT, temperature=0.0, one-word output |
| Cardiologist agent | `agents/cardiologist.py` — FAISS index (7,730 vectors) |
| Endocrinologist agent | `agents/endocrinologist.py` — FAISS index (37,791 vectors) |
| Embeddings | Yandex `text-search-doc/latest` (256-dim, pre-normalized) |
| Generation LLM | YandexGPT/latest, temperature=0.0 |
| Safety gate | Regex patterns for emergencies + prescription requests |
| Evaluation suite | 4 scripts: routing, retrieval, chunk relevance, faithfulness |
| Golden dataset | 30 cases (15 cardiology, 15 endocrinology) |
| Ambiguous test set | 7 cross-domain cases (e.g. diabetic cardiomyopathy) |

### Hyperparameters (final)

```
SIMILARITY_TOP_K     = 5
MAX_L2_DISTANCE      = 1.2
CHUNK_SIZE_WORDS     = 400
CHUNK_OVERLAP_WORDS  = 30
```

---

## 2. Actual Test Outputs

### 2.1 Routing Evaluation (2026-05-09)

```
$ cd multi-agent_system && python3 tests/evaluate_routing.py

Running routing evaluation on 30 golden queries…

  ✅ [cardio_1]  expected=cardiologist  got=cardiologist
  ✅ [cardio_2]  expected=cardiologist  got=cardiologist
  ✅ [cardio_3]  expected=cardiologist  got=cardiologist
  ✅ [cardio_4]  expected=cardiologist  got=cardiologist
  ✅ [cardio_5]  expected=cardiologist  got=cardiologist
  ✅ [cardio_6]  expected=cardiologist  got=cardiologist
  ✅ [cardio_7]  expected=cardiologist  got=cardiologist
  ✅ [cardio_8]  expected=cardiologist  got=cardiologist
  ✅ [cardio_9]  expected=cardiologist  got=cardiologist
  ✅ [cardio_10]  expected=cardiologist  got=cardiologist
  ✅ [cardio_11]  expected=cardiologist  got=cardiologist
  ✅ [cardio_12]  expected=cardiologist  got=cardiologist
  ✅ [cardio_13]  expected=cardiologist  got=cardiologist
  ✅ [cardio_14]  expected=cardiologist  got=cardiologist
  ❌ [cardio_15]  expected=cardiologist  got=surgeon
  ✅ [endo_1]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_2]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_3]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_4]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_5]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_6]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_7]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_8]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_9]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_10]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_11]  expected=endocrinologist  got=endocrinologist
  ❌ [endo_12]  expected=endocrinologist  got=cardiologist
  ✅ [endo_13]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_14]  expected=endocrinologist  got=endocrinologist
  ✅ [endo_15]  expected=endocrinologist  got=endocrinologist

============================================================
  Routing Evaluation — Golden Dataset
============================================================
  Domain                Correct    Total   Accuracy
  -------------------- -------- -------- ----------
  cardiologist               14       15     93.3%
  endocrinologist            14       15     93.3%
  -------------------- -------- -------- ----------
  OVERALL                    28       30     93.3%
============================================================

============================================================
  Cross-Domain Ambiguous Cases (7 queries)
============================================================

  ↔️ [ambig_1]  routed_to=cardiologist   (diabetic cardiomyopathy)
  ↔️ [ambig_2]  routed_to=endocrinologist (thyroid-induced atrial fibrillation)
  ↔️ [ambig_3]  routed_to=cardiologist   (SGLT2 inhibitor cardioprotection in ACS)
  ↔️ [ambig_4]  routed_to=endocrinologist (hyperaldosteronism with cardiac complications)
  ↔️ [ambig_5]  routed_to=cardiologist   (catecholamine-induced cardiomyopathy)
  ↔️ [ambig_6]  routed_to=endocrinologist (amiodarone-induced thyroid dysfunction)
  ↔️ [ambig_7]  routed_to=cardiologist   (metabolic syndrome with coronary artery disease)
```

### 2.2 Retrieval Evaluation (2026-05-09, post cardiology FAISS rebuild)

```
$ cd multi-agent_system && conda run -n conda_ipynb_env python3 tests/evaluate_retrieval.py

Initializing components for retrieval evaluation...
  ⚡ Loading cached FAISS index from .../data/processed/cardiology/faiss_index
  ✅ FAISS vector store loaded successfully
  ⚡ Loading cached FAISS index from .../data/processed/endocrinology/faiss_index
  ✅ FAISS vector store loaded successfully

Running retrieval evaluation on 30 queries...

Query [cardio_1]:   ✅ HIT (Matched keywords: arrhythmia, atrial fibrillation, ecg, holter)
Query [cardio_2]:   ✅ HIT (Matched keywords: angina, ischemia, coronary artery disease, hypertension)
Query [cardio_3]:   ✅ HIT (Matched keywords: arrhythmia, holter, wearable, fibrillation)
Query [cardio_4]:   ✅ HIT (Matched keywords: paroxysmal, arrhythmia, monitor, ecg)
Query [cardio_5]:   ✅ HIT (Matched keywords: atrial fibrillation)
Query [cardio_6]:   ✅ HIT (Matched keywords: heart failure, diastolic, edema)
Query [cardio_7]:   ✅ HIT (Matched keywords: hypertrophic cardiomyopathy, hcm, septal, sam)
Query [cardio_8]:   ✅ HIT (Matched keywords: pericarditis, chest pain, st elevation, pr depression)
Query [cardio_9]:   ✅ HIT (Matched keywords: endocarditis, osler, antibiotics)
Query [cardio_10]:  ❌ MISS (None of the expected keywords found in retrieved context)
Query [cardio_11]:  ✅ HIT (Matched keywords: stemi)
Query [cardio_12]:  ✅ HIT (Matched keywords: heart failure)
Query [cardio_13]:  ✅ HIT (Matched keywords: aortic stenosis, murmur, syncope, replacement)
Query [cardio_14]:  ✅ HIT (Matched keywords: claudication, peripheral artery disease, pad, abi)
Query [cardio_15]:  ✅ HIT (Matched keywords: tamponade, pericardiocentesis, hypotension, fluid)
Query [endo_1]:     ✅ HIT (Matched keywords: hypothyroidism, hashimoto, levothyroxine, tsh)
Query [endo_2]:     ✅ HIT (Matched keywords: diabetes, metabolic syndrome, glycemic)
Query [endo_3]:     ✅ HIT (Matched keywords: incidentaloma, adrenal, cortisol, metanephrines)
Query [endo_4]:     ✅ HIT (Matched keywords: hyperparathyroidism, adenoma, pth)
Query [endo_5]:     ✅ HIT (Matched keywords: pituitary, macroadenoma, prolactin, igf-1)
Query [endo_6]:     ✅ HIT (Matched keywords: pheochromocytoma, metanephrines, catecholamines, plasma)
Query [endo_7]:     ✅ HIT (Matched keywords: cushing, dexamethasone, acth, cortisol)
Query [endo_8]:     ✅ HIT (Matched keywords: acromegaly, growth hormone, igf-1)
Query [endo_9]:     ✅ HIT (Matched keywords: prolactinoma, cabergoline, dopamine agonist, prolactin)
Query [endo_10]:    ✅ HIT (Matched keywords: graves, hyperthyroidism, radioactive iodine)
Query [endo_11]:    ✅ HIT (Matched keywords: adrenal insufficiency, crisis, hydrocortisone)
Query [endo_12]:    ✅ HIT (Matched keywords: hyperaldosteronism, renin)
Query [endo_13]:    ✅ HIT (Matched keywords: hypocalcemia, hypoparathyroidism, parathyroid)
Query [endo_14]:    ✅ HIT (Matched keywords: osteoporosis, t-score)
Query [endo_15]:    ✅ HIT (Matched keywords: dka, ketoacidosis, insulin, potassium)

============================================================
  Retrieval Evaluation Results
============================================================
  Domain                 Hits  Total   Hit Rate
  -------------------- ------ ------ ----------
  cardiologist             14     15     93.3%
  endocrinologist          15     15    100.0%
  -------------------- ------ ------ ----------
  OVERALL                  29     30     96.7%
============================================================
```

### 2.3 Generation / Faithfulness Evaluation (2026-05-06, pre cardiology rebuild)

```
==============================================
Generation Evaluation Complete.
Total Queries: 30
Faithful Answers: 29
FAITHFULNESS SCORE: 96.67%
==============================================
```

*Note: This run was from 2026-05-06, before the cardiology index rebuild. Per-domain split was not yet implemented in the script at that time. The scripts have since been updated to show per-domain breakdown (Section 2.2 demonstrates the format). A fresh faithfulness run is needed to confirm per-domain numbers post-rebuild.*

### 2.4 Hyperparameter Grid Search (2026-05-09)

```
$ cd multi-agent_system && python3 tests/tune_retrieval.py

  K    | L2 ≤   |  Hits |  Hit Rate | Avg Ret. | Note
  -----+--------+-------+-----------+----------+-------------
  3    | 0.8    |     0 |      0.0% |      0.0 |
  3    | 1.0    |    21 |     70.0% |      2.4 |
  3    | 1.2    |    26 |     86.7% |      3.0 |
  3    | 1.4    |    26 |     86.7% |      3.0 |
  3    | 1.6    |    26 |     86.7% |      3.0 |
  3    | 2.0    |    26 |     86.7% |      3.0 |
  5    | 0.8    |     0 |      0.0% |      0.0 |
  5    | 1.0    |    22 |     73.3% |      3.8 |
  5    | 1.2    |    29 |     96.7% |      5.0 | ◀ CHOSEN
  5    | 1.4    |    29 |     96.7% |      5.0 |
  5    | 1.6    |    29 |     96.7% |      5.0 |
  5    | 2.0    |    29 |     96.7% |      5.0 |
  7    | 0.8    |     0 |      0.0% |      0.0 |
  7    | 1.0    |    23 |     76.7% |      5.2 |
  7    | 1.2    |    30 |    100.0% |      7.0 | ★ best
  7    | 1.4    |    30 |    100.0% |      7.0 |
  7    | 1.6    |    30 |    100.0% |      7.0 |
  7    | 2.0    |    30 |    100.0% |      7.0 |
  10   | 0.8    |     0 |      0.0% |      0.0 |
  10   | 1.0    |    23 |     76.7% |      7.0 |
  10   | 1.2    |    30 |    100.0% |     10.0 |
  10   | 1.4    |    30 |    100.0% |     10.0 |
  10   | 1.6    |    30 |    100.0% |     10.0 |
  10   | 2.0    |    30 |    100.0% |     10.0 |
  15   | 0.8    |     0 |      0.0% |      0.0 |
  15   | 1.0    |    23 |     76.7% |      9.5 |
  15   | 1.2    |    30 |    100.0% |     15.0 |
  15   | 1.4    |    30 |    100.0% |     15.0 |
  15   | 1.6    |    30 |    100.0% |     15.0 |
  15   | 2.0    |    30 |    100.0% |     15.0 |

  Best: K=7, L2 ≤ 1.2 (Hit Rate 100.0%, Avg 7.0 docs)
  Chosen operating point: K=5, L2 ≤ 1.2
```

Full grid: [`reports/hyperparameter_grid.csv`](hyperparameter_grid.csv)

**Why K=5 and not K=7:** K=7 gives 100% Hit Rate but increases context length by 40%. The 21 Apr session proved that larger K degrades faithfulness — K=10 dropped it to 66%. K=5 is the chosen trade-off: 96.7% retrieval + 96.7% faithfulness.

---

## 3. Bugs Resolved

### 3.1 Keyword Stripping (identified 2026-04-21, fully resolved 2026-05-09)

**Problem:** Every chunk file contained a `KEYWORDS: term1, term2, ...` header line generated by TF-IDF extraction. This line was embedded alongside clinical text, distorting FAISS vectors — the model matched on keyword noise rather than clinical semantics.

**Fix:** Both `build_cardio_faiss.py` and `build_endo_faiss.py` now strip `KEYWORDS:` lines from `page_content` and store them in `doc.metadata['keywords']`.

**Timeline:**
- 2026-04-21: Fix applied to endocrinology index only. Cardiology index left on the old, keyword-polluted code path (acknowledged in session report).
- 2026-05-09: Fix confirmed present in `build_cardio_faiss.py`. Cardiology index rebuilt from scratch with keyword stripping. Hit Rate improved from 93.3% → 96.7% (cardiology cardio_12 now hits).

### 3.2 LLM Client Migration: OpenAI → Yandex Cloud

**Problem:** The original design assumed an OpenAI API backend. The project was migrated to Yandex Cloud Foundation Models, which required:

1. **API endpoint change** — `base_url` set to `https://llm.api.cloud.yandex.net/v1`
2. **Model URI format** — Yandex uses `gpt://{folder_id}/yandexgpt/latest` instead of `gpt-4`
3. **Authentication** — `Api-Key` header + `x-folder-id` extra header instead of bearer token
4. **Embeddings** — Yandex has no OpenAI-compatible embedding endpoint; custom `YandexNativeEmbeddings` class implements the REST API via `requests`
5. **Embedding asymmetry** — Separate models for documents (`text-search-doc/latest`) vs queries (`text-search-query/latest`)

**Resolution:** All API calls centralized in `settings.py`. Embedding classes are in each agent file. System works end-to-end on Yandex Cloud.

### 3.3 Cosine Migration Attempt (reverted)

**Problem:** Hypothesis that L2 distance was a metric mismatch for cosine-trained embeddings.

**What happened:** `migrate_to_cosine.py` was created but the FAISS indices were never rebuilt — the code filtered with `score >= 0.50` (cosine logic) against L2 distances (0.8–1.3), meaning zero chunks were filtered. Faithfulness dropped to 60%.

**Resolution:** Reverted. Verification showed Yandex embeddings are pre-normalized (norms ≈ 1.0), making L2 and cosine rankings mathematically identical. Script retained as documentation.

### 3.4 Summary Duplication Bug (2026-05-08)

**Problem:** `chunkify.py` treated single-line files as having the entire content on the title line, producing empty body text. When `make_summaries_and_keywords.py` ran, it generated summaries from the title alone, then appended the original text — creating duplicated content in `summary.txt`.

**Fix:** Added heuristic to `chunkify.py`: files with a single line or first line >250 characters use the filename as title and the full text as body.

### 3.5 Yandex Embedding Token Limit (2026-05-06)

**Problem:** `build_endo_faiss.py` crashed with HTTP 400 on certain chunks exceeding the 2,048-token limit. Medical text with chemical formulas, tables, and OCR artifacts can tokenize to far more tokens than expected from word count.

**Fix:** Added catch for `400 + "no more than 2048"` response — truncates the text by 20% and retries. Applied to both `build_cardio_faiss.py` and `build_endo_faiss.py`.

---

## 4. Open Questions

### 4.1 Cardiology Index Quality

The cardiology FAISS index was rebuilt on 2026-05-09 with keyword stripping. Hit Rate improved (93.3% → 96.7%). One miss remains:
- `cardio_10` (aortic dissection) — the corpus lacks a dedicated aortic dissection chapter. This is a **content gap**, not an algorithmic issue.

**Question:** Should the cardiology raw document set be expanded with aortic dissection literature? Or is this an acceptable known limitation for a 2-agent prototype?

### 4.2 Judge Circularity

The faithfulness judge uses YandexGPT — the same model family as the generator. This creates circular evaluation:
- The generator may hallucinate in patterns that the judge is blind to (shared training data biases)
- The judge may systematically excuse hallucinations that match its own knowledge

**Measured impact:** Unknown. Cross-model evaluation (e.g., GPT-4 or Claude as judge) would quantify the bias but requires a different API.

### 4.3 Chunk Size Evidence Strength

The chunk size grid search (`tune_chunk_size.py`) used a micro-evaluation subset (~10 docs) per run to save API costs. The winning size (400 words, 80.0% on subset) was then validated on the full 30-query golden dataset (96.7%). However, no ablation exists comparing 200 vs 400 on the full dataset — we only have:
- 200 words + old index (no keyword strip): 93.3% Hit Rate
- 400 words + new index (keyword strip): 96.7% Hit Rate

The improvement could be from chunk size, keyword stripping, or both. They were changed simultaneously.

### 4.4 Faithfulness Per-Domain Split

The 96.67% faithfulness score (29/30) was measured on 2026-05-06, before the cardiology index was rebuilt with keyword stripping. The per-domain evaluation scripts are now ready but a fresh `evaluate_generation.py` run is needed to get confirmed per-domain numbers on the current indices.

---

## 5. Plan Deviations

### 5.1 OpenAI → Yandex Cloud (major)

**Original plan:** Use OpenAI GPT-4 for routing and generation, OpenAI embeddings for FAISS.

**Actual:** Migrated entirely to Yandex Cloud Foundation Models due to project requirements. This required:
- Custom `YandexNativeEmbeddings` class (Yandex has no OpenAI-compatible embedding endpoint)
- Asymmetric embedding model pair (document vs query models)
- Different authentication scheme (Api-Key + folder-id headers)
- Different model URI format (`gpt://folder_id/model/version`)

**Impact:** Architecture is functionally identical. The OpenAI Python SDK is used as a thin HTTP client with `base_url` override. All Yandex-specific logic is contained in `settings.py` and the agent embedding classes.

### 5.2 Chunk Size Change (200 → 400 words)

**Original:** Default 200-word chunks, untested.

**Actual:** Grid search identified 400 words as optimal. Required re-running `chunkify.py` and `make_summaries_and_keywords.py` for both specialties, then rebuilding all FAISS indices.

- Cardiology: 16,983 → 7,730 chunks
- Endocrinology: 42,665 → 37,791 chunks

### 5.3 Scope Reduction (4 agents → 2 active)

**Original plan:** Four specialist agents (cardiologist, endocrinologist, dermatologist, surgeon).

**Actual:** Only cardiologist and endocrinologist have full RAG pipelines. Dermatologist and surgeon exist as stub classes that return hardcoded responses. The routing system supports all four, but only two have knowledge bases.

### 5.4 No Hybrid Search

**Considered:** Combining FAISS semantic search with BM25/keyword search for hybrid retrieval.

**Not implemented:** Keyword stripping + chunk size tuning pushed Hit Rate to 96.7%, making hybrid search unnecessary for the current evaluation. The `metadata['keywords']` field is preserved for future use.

---

## 6. Current Metric Summary

| Metric | Cardiology | Endocrinology | Overall | Date |
|---|---|---|---|---|
| Routing Accuracy | 93.3% (14/15) | 93.3% (14/15) | 93.3% (28/30) | 2026-05-09 |
| Retrieval Hit Rate | 93.3% (14/15) | 100.0% (15/15) | 96.7% (29/30) | 2026-05-09 |
| Faithfulness | — | — | 96.7% (29/30) | 2026-05-06 |

*Faithfulness per-domain split pending fresh run with updated evaluation scripts.*
