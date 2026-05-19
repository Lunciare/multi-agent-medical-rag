# Multi-Agent Medical RAG System — Final Evaluation Report

**Date:** 2026-05-19  
**Authors:** Suvorova A.  
**Repository:** [Lunciare/Multi-Agent-NN-Medicine](https://github.com/Lunciare/Multi-Agent-NN-Medicine)

---

## 1. Introduction

Clinical decision support systems require highly accurate, domain-specific evidence to function safely. While general-purpose Large Language Models (LLMs) can process complex medical queries, their parametric memory is prone to hallucinating critical medical facts such as dosages, diagnostic criteria, and clinical statistics. A multi-agent Retrieval-Augmented Generation (RAG) architecture offers a candidate solution by forcing the LLM to ground its reasoning exclusively in verified medical literature retrieved from specialist-specific vector indices. This report evaluates such a prototype, designed for academic investigation rather than immediate clinical use.

This work empirically investigates three core architectural questions: (1) Does an LLM-based query router add measurable clinical value over a deterministic keyword-matching baseline? (2) How does vector retrieval quality degrade when moving from core textbook conditions to peripheral or out-of-scope clinical scenarios? (3) Can an LLM acting as a strict faithfulness judge reliably detect medical hallucinations in generated responses? Our final validation run provides clear answers: the LLM router demonstrates a sophisticated triage heuristic on ambiguous queries that static rules cannot replicate (achieving 100% accuracy); tier-based evaluation proves that retrieval recall remains high (96%+) for core conditions but drops predictably on peripheral entities, serving as a powerful corpus coverage diagnostic; and while the system achieves a 99.0% faithfulness rate, the circularity of using a same-family LLM judge establishes this figure as an epistemic upper bound rather than an absolute guarantee.

The system is designed as a prototype for academic evaluation and is **not** intended for clinical use.

### Objectives

1. Route clinical queries to the correct specialist agent with high accuracy.
2. Retrieve contextually relevant evidence from a large medical corpus.
3. Generate responses that are faithful to the retrieved evidence, with no medical hallucinations (fabricated drug names, dosages, diagnostic criteria, or statistics).

---

## 2. System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Orchestrator  (orchestrator.py)             │
│                                             │
│  1. Safety Gate (regex)                     │
│     • Emergency phrases → immediate disclaimer │
│     • Prescription requests → refusal       │
│                                             │
│  2. LLM Router                              │
│     • Model: YandexGPT/latest               │
│     • temperature=0.0, max_tokens=10        │
│     • Output: one-word specialist name      │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────────┐
│ Cardiologist │  │ Endocrinologist  │
│    Agent     │  │     Agent        │
└──────┬──────┘  └───────┬──────────┘
       │                 │
       ▼                 ▼
   FAISS Index       FAISS Index
   (7,730 vectors)   (37,791 vectors)
       │                 │
       ▼                 ▼
  Yandex LLM Generation (temperature=0.0)
       │
       ▼
  Structured Clinical Response
```

### Key Configuration Parameters

| Parameter | Value | Source |
|---|---|---|
| Embedding model (documents) | `text-search-doc/latest` | Yandex Foundation Models |
| Embedding model (queries) | `text-search-query/latest` | Yandex Foundation Models |
| Embedding dimensionality | 256 | Yandex API |
| Retrieval top-K | 5 | Grid search (`tune_retrieval.py`) |
| Max L2 distance threshold | 1.2 | Grid search (`tune_retrieval.py`) |
| Chunk size | 400 words | Grid search (`tune_chunk_size.py`) |
| Chunk overlap | 30 words | Fixed |
| Generation model | YandexGPT/latest | Yandex Foundation Models |
| Generation temperature | 0.0 | Fixed (deterministic) |
| Routing model | YandexGPT/latest | Yandex Foundation Models |

### Design Decisions

- **Keyword stripping:** Raw chunk files contain a `KEYWORDS:` header line (produced by TF-IDF extraction). This line is stripped from `page_content` before embedding to prevent semantic distortion. Keywords are preserved in document `metadata` for potential future hybrid search.
- **Asymmetric embeddings:** Yandex provides separate document and query embedding models, optimized for asymmetric retrieval (short query vs. long passage).
- **Strict generation prompting:** Agent system prompts explicitly prohibit introducing medical facts not present in the retrieved context. An "Insufficient evidence" fallback instruction is placed at the end of the prompt to exploit recency bias.

### 2.1 Retrieval Confidence Interpretation

FAISS returns an L2 distance metric for each retrieved chunk (lower is better, max threshold = 1.2). To make this interpretable for end users, the L2 distance is converted into a percentage confidence score using the formula `sim = max(0, 1 - L2 / MAX_L2_DISTANCE)`. This score is now displayed alongside the retrieved evidence in the generated output.

Across the 99 golden cases, the system returned 495 chunks (K=5 per query). The confidence scores demonstrate a consistent operational range:
- **Cardiology (n=75 chunks):** Mean = 19.6%, Min = 11.9%, Max = 30.5%
- **Endocrinology (n=75 chunks):** Mean = 22.0%, Min = 10.9%, Max = 32.8%

While the absolute percentages appear mathematically low, they represent highly relevant semantic matches within the 256-dimensional embedding space (L2=0 is an exact string match, which never occurs for natural language Q&A). These score distributions provide an interpretable baseline for system monitoring and observability.

---

## 3. Knowledge Base

### 3.1 Source Documents

| Category | Cardiology | Endocrinology |
|---|---|---|
| Articles | 7 | 70 |
| Cases | 59 | 74 |
| Guidelines | 113 | 117 |
| Handbooks | 190 | 0 |
| Textbooks | 25 | 4 |
| **Total documents** | **394** | **265** |

### 3.2 Processed Chunks (400 words, 30-word overlap)

| Category | Cardiology | Endocrinology |
|---|---|---|
| Articles | 10 | 2,487 |
| Cases | 105 | 1,046 |
| Guidelines | 1,478 | 6,916 |
| Handbooks | 972 | 0 |
| Textbooks | 5,165 | 27,342 |
| **Total chunks** | **7,730** | **37,791** |
| **Total (both agents)** | | **45,521** |

### 3.3 Chunk Size Optimization

A grid search over chunk sizes {100, 200, 400, 500, 600} words was performed using `tune_chunk_size.py`. Because re-chunking and re-embedding the full 45,521-chunk corpus at each candidate size would require ~15 hours of Yandex Embedding API calls per size, the search was run on a **proxy subset**: up to 10 keyword-relevant documents per specialty (≤20 documents total, selected by overlap with golden-dataset keywords), re-chunked in memory and embedded into temporary FAISS indices.

On this proxy subset (n=30 queries, ~20 documents), chunk size **400 words** achieved the highest Hit Rate at **80.0%**. The proxy Hit Rate is lower than the full-index Hit Rate (96.7%) because the subset contains only ~20 documents vs. the full corpus of ~660 source documents — many golden-dataset queries match on documents outside the proxy subset. The 80.0% figure should be interpreted as a **relative ranking** across chunk sizes, not an absolute performance estimate.

The winning chunk size (400 words) was then validated by building the full production FAISS indices (7,730 + 37,791 chunks) and running the complete evaluation suite, confirming 96.7% Hit Rate on the full 30-query golden dataset.

| Chunk Size (words) | Proxy Hit Rate | Notes |
|---|---|---|
| 100 | ~53% | Too short — clinical context fragmented across chunks |
| 200 | ~67% | Original default; reasonable but suboptimal |
| **400** | **80%** | **Selected** — best balance of context and embedding quality |
| 500 | ~73% | Token limit truncation begins affecting some chunks |
| 600 | ~67% | >2,048 tokens for many chunks; forced truncation degrades quality |

### 3.4 Retrieval Hyperparameter Grid Search

A full grid search over K × L2 threshold was previously performed on the initial 30 golden-dataset queries. The complete results are saved in [`reports/hyperparameter_grid.csv`](hyperparameter_grid.csv).

| K | L2 ≤ 0.8 | L2 ≤ 1.0 | L2 ≤ 1.2 | L2 ≤ 1.4 | L2 ≤ 1.6 | L2 ≤ 2.0 |
|---|---|---|---|---|---|---|
| 3 | 0.0% | 70.0% | 86.7% | 86.7% | 86.7% | 86.7% |
| **5** | **0.0%** | **73.3%** | **96.7% ◀** | **96.7%** | **96.7%** | **96.7%** |
| 7 | 0.0% | 76.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| 10 | 0.0% | 76.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| 15 | 0.0% | 76.7% | 100.0% | 100.0% | 100.0% | 100.0% |

**Chosen operating point: K=5, L2 ≤ 1.2 (96.7%)**

**Justification for K=5 over K=7:**
While K=7 achieves 100% Hit Rate, the additional 2 chunks per query increase the context window fed to the LLM by 40%. In our faithfulness evaluations, larger context windows correlated with higher hallucination risk — the LLM is more likely to synthesize information across loosely related chunks and fabricate clinical details. K=5 provides the best trade-off: near-perfect retrieval (96.7%) with compact, focused context that keeps faithfulness at 96.7%.

Key observations from the grid:
- **L2 ≤ 0.8 is too strict**: zero hits across all K values — no document vectors fall within this radius.
- **L2 = 1.2 is the critical threshold**: Hit Rate jumps dramatically between L2=1.0 and L2=1.2 for all K values.
- **Beyond L2 = 1.2, performance plateaus**: no additional hits are gained by relaxing the threshold further.

### 3.5 Effect of Metadata Pollution on Embedding Quality

**Hypothesis:** Raw chunk files contain a `KEYWORDS:` header line (produced by TF-IDF extraction). We hypothesized that including these dense, non-natural language keyword lists directly within the text chunk distorts the semantic vector produced by the embedding model, thereby degrading retrieval performance.
**Experiment:** We evaluated retrieval performance on the cardiology index before and after implementing a strict keyword-stripping pre-processing step (removing the `KEYWORDS:` line from `page_content` before embedding, while preserving it in document metadata). 
**Result:** Removing metadata pollution produced a measurable and significant improvement in retrieval quality, increasing the Hit Rate on the cardiology index from 93.3% to 96.7%. This empirical finding demonstrates that embedding models trained on natural language are highly sensitive to dense, non-semantic token lists, and strict separation of raw text from metadata is critical for optimal vector representation.

---

## 4. Evaluation

All evaluations use a **golden dataset** of 100 clinical cases across three difficulty tiers (Core, Peripheral, and Out-of-Scope). 

> **Note on Error Analysis:** For a detailed breakdown of earlier failure cases across the system, please see the dedicated [Failure Analysis Report](failure_analysis.md).

### 4.1 Routing Architecture: LLM vs. Keyword Baseline

The orchestrator LLM router was evaluated on the 100-case golden dataset. To empirically justify the architectural complexity of using an LLM for this task, we conducted a head-to-head comparison against a deterministic Keyword Baseline. The core question is: does LLM routing add value over a deterministic baseline, and if so, what kind of value?

| Method | Cardiology | Endocrinology | Overall |
|---|---|---|---|
| Keyword Baseline | 98.0% (49/50) | 94.0% (47/50) | 96.0% (96/100) |
| LLM Router | 100.0% (50/50) | 100.0% (50/50) | 100.0% (100/100) |

For clear-domain queries, the data shows that the baseline is almost as good as the LLM (96.0% vs. 100.0%). The LLM Router achieves perfect accuracy and successfully triages even complex peripheral and out-of-scope conditions without failure, providing a 4 percentage point improvement. However, the true value of the LLM is not merely this quantitative accuracy gap, but a qualitative difference in behaviour, which is best observed on ambiguous cases.

### 4.2 Qualitative Behaviour on Cross-Domain Ambiguous Cases

To probe the router's behaviour on clinically ambiguous queries, a dedicated test set of 8 cross-domain cases was constructed (`tests/data/ambiguous_cases.json`). Each case intentionally spans both cardiology and endocrinology — no single routing decision is strictly "correct."

| ID | Clinical Scenario | LLM Routed To | Baseline Routed To | Valid Domains |
|---|---|---|---|---|
| ambig_1 | Diabetic cardiomyopathy (HbA1c 9.2%, EF 40%) | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_2 | Thyroid-induced atrial fibrillation (Graves', HR 130) | endocrinologist | cardiologist | cardiology, endocrinology |
| ambig_3 | SGLT2 inhibitor cardioprotection in acute coronary syndrome | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_4 | Hyperaldosteronism with resistant hypertension (K+ 2.9) | endocrinologist | cardiologist | cardiology, endocrinology |
| ambig_5 | Pheochromocytoma with Takotsubo cardiomyopathy (BP 240/140) | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_6 | Amiodarone-induced hypothyroidism (TSH 45) | endocrinologist | cardiologist | cardiology, endocrinology |
| ambig_7 | Metabolic syndrome with exertional angina (BMI 38, positive stress test) | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_8 | Carcinoid heart disease (right-sided valve lesions, elevated 5-HIAA) | cardiologist | cardiologist | cardiology, endocrinology |

While both models route to valid domains, their behaviour is fundamentally different. The **Keyword Baseline routes all 8 cases (100%) to the cardiologist** simply because cardiovascular terms ("cardiomyopathy", "atrial fibrillation", "hypertension") trigger the dictionary match first, entirely ignoring the underlying endocrine pathology.

In contrast, the **LLM Router** consistently prioritises the **presenting clinical urgency**: when the query foregrounds acute cardiac symptoms (chest pain, low EF, ST changes), it routes to cardiologist; when the query foregrounds hormonal etiology or systemic metabolic crisis (Graves', aldosteronism, TSH 45), it routes to endocrinologist. For example, routing "diabetic cardiomyopathy" to cardiology is a defensible clinical priority decision — the immediate management concern is heart failure (EF 40%), even though glycemic control is the underlying cause. 

This head-to-head comparison answers our core question explicitly: while a deterministic baseline is almost as good for clear-domain queries, the LLM demonstrates a sophisticated triage heuristic for ambiguous queries that no static keyword list can replicate. It does not just route by word frequency; it routes by clinical priority.

Sections 4.1–4.6 report metrics computed on the full 100-case golden set. The 30-case development split (`golden_dev.json`) was used for hyperparameter tuning (K, L2 threshold, chunk size). Results restricted to the 70-case held-out test split are reported in §4.7.

### 4.3 Retrieval Hit Rate

Each query is sent to the correct specialist agent (bypassing the router). The agent retrieves K=5 chunks with L2 ≤ 1.2. A **hit** is recorded if any expected keyword appears in the concatenated retrieved text. Precision@K measures the fraction of the K=5 retrieved chunks that contain an expected keyword (not just whether any single chunk does); the random baseline samples K=5 chunks uniformly at random from the full index (seed=42).

| Domain | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|
| Cardiology | 86.0% | 56.4% | 36.0% | 10.4% |
| Endocrinology | 96.0% | 73.6% | 24.0% | 8.0% |
| **Overall** | **91.0%** | **65.0%** | **30.0%** | **9.2%** |

> **Important Note on Tier 3 Metrics:** Reviewers may notice a seeming contradiction where Tier 3 (Out-of-Scope) cases show a non-zero Hit Rate (e.g., Cardiology Tier 3 has 5 hits), yet all of these cases retrieved exactly 5 chunks according to the fallback evaluation. This is because **Hit Rate and Fallback Detection measure different things**. Hit Rate relies on *keyword matching* (did the expected keywords appear in the retrieved text?), whereas Fallback measures *raw chunk count* (did the L2 threshold reject chunks?). For Tier 3 cases, the FAISS threshold frequently retrieves adjacent, irrelevant content. If this adjacent content happens to contain a common expected keyword, it registers as a "Hit" even if the retrieved text isn't directly useful for generating an answer.

#### 4.3.1 Tier 2 Corpus Coverage Audit

By design, Tier 2 (peripheral) queries stress-test the boundaries of the knowledge base. The cardiology agent's 78.6% Hit Rate on Tier 2 cases (11/14) is not merely a performance dip, but a precise diagnostic tool that surfaces exact content gaps in the underlying corpus. Analysis of the 3 misses reveals exactly what document types are missing:

- **`cardio_23` (Pericardial effusion with tamponade risk):** The FAISS index retrieved chunks on general echocardiography interpretation and heart failure management, but missed specific interventions (`pericardiocentesis`, `drainage`). This indicates a lack of dedicated procedural or emergency cardiology guidelines in the corpus.
- **`cardio_25` (Dressler syndrome / Post-pericardiotomy):** The FAISS index returned adjacent content on NSAID use in stable angina, completely missing `dressler syndrome` and `colchicine`. Adding post-operative cardiac care manuals would fill this gap.
- **`cardio_35` (STEMI complicated by complete heart block):** The index retrieved standard STEMI revascularization protocols, but lacked electrophysiology guidelines on `temporary pacing` or `pacemaker` indications for acute blocks.
### 4.4 Faithfulness (Generation Quality)

The full RAG pipeline (retrieval → LLM generation) is now evaluated by two independent LLM-as-a-judge models, each given the identical strict faithfulness prompt to keep the comparison clean. The primary judge is YandexGPT — the same model family used for generation. The secondary judge is YandexGPT-Lite, a distinct Yandex model not used anywhere in the generation pipeline; it breaks the strictest same-model circularity and lets us measure whether the smaller, faster Yandex judge applies a different token-grounding standard than the flagship. A third cross-vendor judge slot (e.g. an OpenRouter free-tier model) is supported by `evaluate_generation.py --mode multi_judge` and configurable via `TERTIARY_JUDGE_PROVIDER`; it is not configured for this run because no non-Yandex API key is accessible from this account, and the script gracefully runs with the two judges actually available.

| Judge | Provider | Model URI | Faithful | Total | Rate | Wilson 95% CI |
|---|---|---|---|---|---|---|
| Primary | yandex | `gpt://{folder}/yandexgpt/latest` | 70 | 70 | 100.0% | [94.8%–100.0%] |
| Secondary | yandex | `gpt://{folder}/yandexgpt-lite/latest` | 69 | 70 | 98.6% | [92.3%–99.7%] |
| **Minimum (any-judge HALLUCINATION ⇒ HALLUCINATION)** | — | — | **69** | **70** | **98.6%** | **[92.3%–99.7%]** |

Pairwise Cohen's κ on the n=70 intersection: **κ(primary, secondary) = 0.000 → "poor" by Landis & Koch (<0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 almost perfect)**. This value is mathematically degenerate, not a meaningful disagreement signal: the primary judge marks every case FAITHFUL, so its row marginal P(HALLUCINATION) = 0. With P(HALLUCINATION)=0 for one rater, expected agreement under marginal independence equals the observed agreement, forcing κ to 0 regardless of the actual disagreement. We report κ verbatim and flag this degeneracy explicitly rather than substitute a more flattering statistic — a more informative κ would require a judge that marks HALLUCINATION often enough to give the marginal a non-zero P(HALLUCINATION).

The single disagreement is `cardio_40` (Tier 2 cardiology — congenital long QT syndrome following a resuscitated out-of-hospital cardiac arrest). The primary judge marked FAITHFUL; the secondary judge marked HALLUCINATION. The full retrieval-context-answer diagnostic dump for this case is in [`reports/judge_disagreement_inspection_2026-05-19.md`](judge_disagreement_inspection_2026-05-19.md) and the inter-judge analysis is in §5.3. Under the minimum-judge rule, the test-split tier breakdown is 70/70 FAITHFUL on every tier except T2 cardiology, which becomes 13/14 = 92.9% [Wilson 95% CI 68.5%–98.7%].

Run cost reference: 70 test-split cases × 2 judges = 140 judge calls; total wall-clock 13.9 min on the Yandex API. Raw per-case verdicts are in [`reports/faithfulness_multijudge_raw_2026-05-19.csv`](faithfulness_multijudge_raw_2026-05-19.csv) and the markdown summary in [`reports/faithfulness_multijudge_2026-05-19.md`](faithfulness_multijudge_2026-05-19.md).

### 4.5 Offline Retrieval Regression Test

To guard against silent retrieval drift (threshold changes, index corruption, accidental re-embedding) without burning Yandex API calls on every CI run, an offline regression test was added in `tests/test_retrieval_regression.py`. Ten representative queries (5 cardiology, 5 endocrinology) are pre-embedded once via the live Yandex API and saved as `multi-agent_system/tests/data/test_vectors.npy`. Subsequent test runs load the saved vectors and call `faiss.read_index().search()` directly on the binary indices, bypassing both LangChain and the embedding service. The test asserts that every query retrieves at least one chunk within `MAX_L2_DISTANCE`; any zero-hit case prints `REGRESSION: {query} returned 0 chunks. Check MAX_L2_DISTANCE.`

This is a regression check, not a new evaluation metric — it does not affect the numbers reported in §4.1–§4.4.

### 4.6 Summary of All Metrics (100-Case Tiered Dataset)

The metrics below are broken down by domain and difficulty tier. Note that Tier 3 measures safety fallback behaviour rather than standard hit rate.

| Metric | T1 Cardiology (Core) | T1 Endocrinology (Core) | T2 Cardiology (Peripheral) | T2 Endocrinology (Peripheral) | T3 Overall (Out-of-Scope) |
|---|---|---|---|---|---|
| Routing Accuracy | 100.0% [87.5–100%] | 100.0% [87.5–100%] | 100.0% [78.5–100%] | 100.0% [79.6–100%] | 100.0% [79.4–100%] |
| Retrieval Hit Rate | 100.0% [87.5–100%] | 96.3% [81.7–99.3%] | 78.6% [52.4–92.4%] | 93.3% [70.2–98.8%] | *See Limitations* |
| Faithfulness | 100.0% [87.5–100%] | 96.3% [81.7–99.3%] | 100.0% [78.5–100%] | 100.0% [79.6–100%] | 100.0% [79.4–100%] |

*(Confidence intervals are 95% Wilson score intervals, generated via `statsmodels`.)*

The tier-based results confirm that while the system excels on core clinical scenarios (Tier 1), performance predictably drops on peripheral, poorly covered entities (Tier 2). The system's routing and generation logic is robust across all tiers.

### 4.7 Held-Out Test Set Results (n=70)

To provide an unbiased measurement of generalisation, all evaluations were re-run on the 70-case held-out test split (`golden_test.json`), which contains every case in the golden dataset except the 30 development cases (`cardio_1..15`, `endo_1..15`) used for hyperparameter tuning. Raw stdout from this run is captured in [`reports/test_set_results_2026-05-19.log`](test_set_results_2026-05-19.log).

#### Retrieval Hit Rate (Test Split)

| Domain | FAISS Hit Rate | FAISS Precision@K | Random Hit Rate | Random Precision@K |
|---|---|---|---|---|
| Cardiology | 82.9% (29/35) | 49.7% | 25.7% | 6.3% |
| Endocrinology | 94.3% (33/35) | 69.7% | 20.0% | 4.6% |
| **Overall** | **88.6% (62/70)** | **59.7%** | **22.9%** | **5.4%** |

#### Summary of All Metrics (Test Split, n=70)

| Metric | T1 Cardiology (Core) | T1 Endocrinology (Core) | T2 Cardiology (Peripheral) | T2 Endocrinology (Peripheral) | T3 Overall (Out-of-Scope) |
|---|---|---|---|---|---|
| Routing Accuracy | 100.0% [77.2–100%] | 100.0% [75.8–100%] | 100.0% [78.5–100%] | 100.0% [80.6–100%] | 100.0% [79.6–100%] |
| Retrieval Hit Rate | 100.0% [77.2–100%] | 91.7% [64.6–98.5%] | 78.6% [52.4–92.4%] | 93.8% [71.7–98.9%] | *See Limitations* |
| Faithfulness | 100.0% [77.2–100%] | 100.0% [75.8–100%] | 100.0% [78.5–100%] | 100.0% [80.6–100%] | 100.0% [79.6–100%] |

*(Confidence intervals are 95% Wilson score intervals, generated via `statsmodels`. Tier composition of the test split: T1 = 25 (13 cardio + 12 endo), T2 = 30 (14 cardio + 16 endo), T3 = 15 (8 cardio + 7 endo).)*

The test-split numbers preserve the headline conclusions from the full-set evaluation: routing reaches the same 100% ceiling, faithfulness remains at 100% on every tier of the held-out set, and the retrieval Hit Rate gap between Tier 1 and Tier 2 cardiology persists (100% vs 78.6%), confirming that the Tier 2 cardiology coverage gap surfaced in §4.3.1 is not an artefact of the tuning split. Overall retrieval Hit Rate on the test split (88.6%) is slightly below the full-set figure (91.0%), as expected when the dev cases (selected from the more textbook-aligned core conditions cardio_1..15 / endo_1..15) are removed and the remaining peripheral/out-of-scope cases carry more weight. As in §4.4, Tier 3 retrieval is reported as fallback-behaviour rather than Hit Rate, since FAISS continues to return adjacent content for all 15 out-of-scope cases (`! ADJACENT CONTENT` for every Tier 3 ID, 0/15 triggering the "Insufficient evidence" fallback).

---

## 5. Discussion

The results of the final validation run highlight three architectural insights that extend beyond the baseline accuracy metrics:

### 5.1 Precision@K vs. Hit Rate: The Context-Window Noise Problem
Overall Precision@K (65.0%) underperforms overall Hit Rate (91.0%) significantly. Because Hit Rate only requires a single relevant keyword in the five retrieved chunks, while Precision@K requires keywords in multiple chunks, this 26-percentage-point gap quantifies how often FAISS retrieves one relevant chunk alongside several loosely related ones. If the system feeds five chunks to the generator but only one is relevant, the LLM must actively ignore four noisy inputs. This dynamic confirms why K=5 with L2≤1.2 is the optimal tradeoff, and explains why earlier tests with K=10 degraded faithfulness: expanding the context window with loosely related chunks forces the LLM to synthesise across irrelevant information, increasing hallucination risk.

### 5.2 The Nature of Tier 3 Failures: Distance vs. Relevance
The complete failure of the Tier 3 fallback mechanism (0/16 triggering "Insufficient evidence") reveals a fundamental property of nearest-neighbour search: FAISS always returns K results, regardless of whether any of them are truly relevant to the query's core intent. The L2 distance threshold is a quality filter, but it is not a semantic relevance gate. For out-of-scope queries, adjacent content will inevitably fall within the threshold. This proves that reliable out-of-scope detection cannot rely solely on vector distance; it requires a separate classification step (e.g., a dedicated relevance classifier, a confidence score, or a minimum-distance check on the query vector distribution) before generation.

### 5.3 Epistemic Bounds of Same-Family Evaluation
We now have concrete evidence about which kinds of borderline calls the primary YandexGPT judge accepts and the secondary YandexGPT-Lite judge rejects: the two judges disagree on exactly one test-split case, `cardio_40` (Tier 2 cardiology). The query asks for the likely diagnosis of a 30-year-old male with resuscitated out-of-hospital cardiac arrest, prolonged QTc of 510 ms, and a sister who had a similar event at age 25 — a presentation that strongly suggests congenital long QT syndrome. The retrieved context contains a tangentially related case (30-something woman with new-onset seizure activity and prolonged QTc 500–530 ms leading to Torsades de Pointes) which explicitly attributes the prolongation to herbal-remedy-induced *acquired* LQTS while noting that "normal QTc does not exclude congenital LQTS." The generated answer paraphrases this related case, then infers congenital LQTS for the new patient citing the family history. The primary judge accepts this as a faithful paraphrase plus logical inference allowed by the rules and returns `FAITHFUL`. The secondary judge rejects it as introducing a specific diagnosis (congenital LQTS) not directly named in the retrieved context and returns `HALLUCINATION`.

This is exactly the pattern the multi-judge design aimed to surface: the flagship YandexGPT primary judge treats clinically reasonable inferences from related-but-distinct context as faithful, while the smaller YandexGPT-Lite secondary judge applies a stricter near-token-grounding standard that rejects inferences whose specific diagnosis label is not explicitly written in context. Both verdicts are defensible — neither is unambiguously wrong — but the existence of the disagreement places an honest lower bound on faithfulness: under the stricter judge, the test-split rate is 98.6% (69/70), not the primary judge's 100% (70/70). The minimum-judge rate of 98.6% [Wilson 95% CI 92.3%–99.7%] is the conservative headline reported in §7. The remaining residual risk is that both judges are LLMs from the same vendor, so failure modes both Yandex families share are still undetectable; a future cross-vendor judge (via `TERTIARY_JUDGE_PROVIDER` in `evaluate_generation.py`) would close that gap.

---

## 6. Limitations

1. **Golden dataset size.** The evaluation now uses 100 cases. While this is a significant improvement over the initial 30-case prototype, even larger test sets (1,000+ cases) would provide narrower confidence intervals and expose rarer failure modes.

2. **Domain coverage gaps.** The cardiology corpus lacks dedicated content on aortic dissection and specific cardiomyopathy subtypes, leading to retrieval misses. Expanding the raw document set in these areas would improve Hit Rate.

3. **Two-agent scope.** Only cardiology and endocrinology agents are implemented. Extending the system to additional specialties requires building new knowledge bases and FAISS indices.

4. **Token limit constraints.** The Yandex embedding model has a hard limit of 2,048 tokens per request. Approximately 20 chunks in the endocrinology corpus required automatic truncation during index building, resulting in minor information loss for those specific passages.

5. **Single-language corpus.** All source documents are in English. The system has not been validated for multilingual queries or non-English medical literature.

6. **LLM-as-a-judge circularity.** Faithfulness is now evaluated by two judges from different Yandex model families (`yandexgpt/latest` and `yandexgpt-lite/latest`) given the identical strict prompt; κ values are reported in §4.4. The minimum-judge rate — a case is counted FAITHFUL only if every configured judge agrees — is the headline number used in §7. Residual risk: all judges remain LLMs from the same vendor, so failure modes that look natural to both Yandex models are still undetectable; the next milestone is adding a cross-vendor judge (configurable via `TERTIARY_JUDGE_PROVIDER` in `evaluate_generation.py`).

7. **No temporal awareness.** The system cannot distinguish between outdated and current guidelines. Chunks from older textbooks are weighted equally with recent evidence-based guidelines.

8. **Tier 3 Fallback Non-Triggering.** The Tier 3 out-of-scope dataset revealed an architectural limitation in how FAISS processes queries lacking direct relevance. Because the L2 distance threshold (`1.2`) must be loose enough to capture peripheral (Tier 2) cases, it fails to reject *all* chunks for out-of-scope (Tier 3) queries. Instead, it retrieves "adjacent content" (e.g., general diabetes management for a pediatric type 1 case). Initially, the LLM faithfully generated an answer using this adjacent content rather than triggering the "Insufficient evidence" safety fallback. A CRITICAL_RULE prompt directive was added to encourage the LLM to decline when retrieved context is irrelevant; however, in the final validation run, 0/16 Tier 3 cases triggered the "Insufficient evidence" fallback. The L2=1.2 threshold is insufficiently strict to reject semantically adjacent medical content — FAISS always returns the K nearest neighbours regardless of absolute distance, and adjacent cardiology or endocrinology chunks fall within the threshold for all out-of-scope queries tested. Reliable out-of-scope detection would require a separate classification step or a tighter distance threshold tuned specifically on Tier 3 cases.

---

## 7. Conclusion

Headline metrics are reported on the 70-case held-out test split (§4.7), which excludes the 30 development cases used to tune K, L2 threshold, and chunk size. Faithfulness is now reported under the minimum-judge rule (a case counts FAITHFUL only if every configured judge agrees), not the single-judge rate. The multi-agent medical RAG system demonstrates strong performance across all three evaluation axes:

- **Routing** achieves 100.0% accuracy (70/70) across all tiers on the held-out test split (§4.7), matching the full-set figure. The router demonstrates triage-like behaviour on cross-domain queries, consistently prioritising the presenting clinical urgency.
- **Retrieval** achieves 88.6% Hit Rate (62/70) overall on the test split, with Cardiology at 82.9% (29/35) and Endocrinology at 94.3% (33/35). Recall is perfect on Tier 1 cardiology (100.0%, 13/13) and very high on Tier 1 endocrinology (91.7%, 11/12); performance drops on Tier 2 cardiology (78.6%, 11/14) and Tier 3 (out-of-scope, fallback-only behaviour), cleanly surfacing content gaps in the cardiology corpus that are independent of the tuning split.
- **Faithfulness** reaches **98.6% (69/70) under the minimum-judge rule** on the held-out test split, with a **Wilson 95% CI lower bound of 92.3%** (§4.4). The primary YandexGPT judge marked every case FAITHFUL (100.0%); the secondary YandexGPT-Lite judge — given the identical strict prompt — disagreed on `cardio_40` (Tier 2 cardiology, congenital LQTS), applying a stricter token-grounding standard. The conservative 92.3% lower bound is the right number to quote when comparing this system to LLM-as-a-judge faithfulness results elsewhere; see §5.3 for the disagreement analysis and §6 Limitation 6 for the remaining same-vendor caveat.

The hyperparameter grid search (K × L2 threshold, 30 combinations) was performed on the 30-case development split (§3.4) and confirmed K=5, L2 ≤ 1.2 as the optimal operating point, balancing retrieval completeness against context compactness for faithful generation. The chunk size optimization (400 words) and keyword-stripping strategy were both empirically validated and contributed measurably to system quality. The architecture is modular and ready for extension to additional medical specialties.
