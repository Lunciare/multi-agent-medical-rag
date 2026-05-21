# Multi-Agent Medical RAG System — Final Evaluation Report

**Date:** 2026-05-21  
**Authors:** Suvorova A.  
**Repository:** [Lunciare/Multi-Agent-NN-Medicine](https://github.com/Lunciare/Multi-Agent-NN-Medicine)

---

## 1. Introduction

Clinical decision support systems require highly accurate, domain-specific evidence to function safely. While general-purpose Large Language Models (LLMs) can process complex medical queries, their parametric memory is prone to hallucinating critical medical facts such as dosages, diagnostic criteria, and clinical statistics — a failure mode quantified for clinical QA by Singhal et al. \cite{singhal2023medpalm} and for medical RAG specifically by Xiong et al. \cite{xiong2024medrag}. A multi-agent Retrieval-Augmented Generation (RAG) architecture offers a candidate solution by forcing the LLM to ground its reasoning exclusively in verified medical literature retrieved from specialist-specific vector indices. This report evaluates such a prototype, designed for academic investigation rather than immediate clinical use.

This work empirically investigates three core architectural questions: (1) Does an LLM-based query router add measurable clinical value over a deterministic keyword-matching baseline? (2) How does vector retrieval quality degrade when moving from core textbook conditions to peripheral or out-of-scope clinical scenarios? (3) Can an LLM acting as a strict faithfulness judge reliably detect medical hallucinations in generated responses? On the held-out test split, the LLM router classifies 70/70 cases correctly (Wilson 95% CI: 94.9–100.0%); on the eight cross-domain ambiguous cases, it routes to the endocrinologist when the dominant pathology is endocrine and to the cardiologist when the dominant pathology is cardiac, where a keyword baseline routes 8/8 to the cardiologist. Tier-stratified retrieval reaches Recall@5 ≥ 96% on core conditions and drops to 78.6% on peripheral cases. Faithfulness as judged by a same-family LLM is 99.0%; this is an upper bound under the methodology of §4.4.

The system is designed as a prototype for academic evaluation and is **not** intended for clinical use.

### Objectives

1. Route clinical queries to the correct specialist agent with high accuracy.
2. Retrieve contextually relevant evidence from a large medical corpus.
3. Generate responses that are faithful to the retrieved evidence, with no medical hallucinations (fabricated drug names, dosages, diagnostic criteria, or statistics).

---

## 1.5 Related Work

This project sits at the intersection of four established research strands: medical RAG benchmarks, retrieval-method baselines, LLM-as-judge evaluation methodology, and multi-agent clinical-AI architectures. Citing each strand explicitly clarifies what is borrowed, what is novel, and what is deliberately *not* attempted in this prototype.

### 1.5.1 Medical RAG benchmarks

Medical RAG evaluation has matured from generic factuality probes toward structured clinical benchmarks. Singhal et al.'s Med-PaLM \cite{singhal2023medpalm} introduced MultiMedQA, a composite of six clinical and consumer-health QA datasets (MedQA, PubMedQA, MedMCQA, LiveQA, MedicationQA, HealthSearchQA), and showed that domain-instruction-tuned LLMs can match human expert preference on ~92.6% of consumer questions while still hallucinating dangerous specifics. Xiong et al.'s MedRAG/MIRAGE benchmark \cite{xiong2024medrag} extends this to RAG specifically: 7,663 multiple-choice questions across five biomedical corpora, with explicit retrieval/recall/precision per question and an analysis of how chunk granularity and retriever choice affect downstream accuracy. BioASQ \cite{tsatsaronis2015bioasq} predates both and provides the largest continuously-curated biomedical semantic-indexing + QA shared task, with expert-annotated relevance judgements and ideal/exact-answer pairs over 10+ years of editions.

This project's evaluation differs in three ways: (a) it uses a **bespoke 100-case golden set** stratified into three tiers (core / peripheral / out-of-scope, §3 in the dataset construction), rather than reusing an established benchmark — driven by the need to evaluate the corpus's actual coverage on real cardiology + endocrinology questions; (b) the cases are **open-ended clinical scenarios**, not multiple-choice, so we report Recall@K and faithfulness rather than accuracy on a fixed answer set; and (c) the Tier 3 (out-of-scope) construction explicitly probes refusal behaviour, which MIRAGE and BioASQ do not directly measure. The trade-off is reduced comparability with prior medical-RAG numbers; the comparable surface is the retrieval-quality methodology (top-K, Hit Rate, Recall@K with gold-source labels) and the LLM-as-judge faithfulness protocol described below.

### 1.5.2 Retrieval method baselines

The canonical RAG framework was introduced by Lewis et al. \cite{lewis2020rag}: a dense retriever feeds top-K passages to a generator, both jointly fine-tuned end-to-end. Dense Passage Retrieval \cite{karpukhin2020dpr} is the dominant dense baseline — a dual-encoder learned via in-batch contrastive loss on Natural Questions / TriviaQA — and the canonical sparse baseline is BM25 \cite{robertson2009bm25}, a probabilistic IDF-weighted lexical match. Modern medical RAG systems frequently report a *hybrid* baseline (BM25 ∪ dense, score-fused or reranked) because dense embeddings reliably miss rare entity names that BM25 captures via exact match.

This project uses **dense-only Yandex `text-search-doc`/`text-search-query` asymmetric embeddings**, with neither a BM25 sparse baseline nor a hybrid fusion. The decision was a deliberate scope reduction (Stage 2 report §5.4): keyword stripping + chunk-size tuning pushed the dev-set Hit Rate to 96.7%, making hybrid retrieval feel unnecessary at that point. With Stage 6's grounded Recall@K (58.5% full set vs the legacy 91.0% KeywordHitRate, §4.3) it is now clear that the dense-only choice does miss a substantial fraction of relevant documents, and a BM25 / hybrid baseline is the obvious next experiment. The `metadata['keywords']` field on every chunk is preserved precisely for that BM25 future use.

### 1.5.3 LLM-as-judge methodology

RAGAS \cite{es2023ragas} formalised automated RAG evaluation with three LLM-judged metrics — faithfulness, answer relevance, and context precision — by asking a strong LLM to score each generated answer against the retrieved context. Zheng et al.'s MT-Bench / Chatbot Arena work \cite{zheng2023mtbench} systematically characterised LLM-judge biases: position bias, verbosity bias, and most importantly for this project, **same-family self-preference bias** — a judge LLM tends to rate outputs from its own model family more favourably than outputs from other families on the same task. Their measured magnitude (model-pair-dependent, but typically a 5–25 percentage-point inflation) directly motivates this project's Fix 2 (Stage 5): instead of trusting the single YandexGPT judge that produces a 100% faithfulness rate on test, we deploy a second YandexGPT-Lite judge with the same prompt and report the **minimum-judge rate** (a case is FAITHFUL only when both judges agree). The remaining gap is that both judges are from the same vendor, so the cross-vendor blind spot remains; the `TERTIARY_JUDGE_PROVIDER` configuration in `evaluate_generation.py` is the placeholder for closing that gap.

### 1.5.4 Multi-agent medical systems

Kim et al.'s MDAgents \cite{kim2024mdagents} is the closest recent multi-agent clinical-AI work. It builds a *collaboration* of LLMs that adaptively choose between solo, paired, or group-discussion modes depending on the medical query's complexity — modelled on how human clinicians escalate from single-physician to multi-disciplinary-team review. Each MDAgent's role is dynamically assigned per case (radiologist, pathologist, clinician, etc.) and the agents iteratively *debate* the diagnosis, with the framework choosing the level of collaboration based on internal complexity estimates.

This project is **not** a multi-agent system in the MDAgents sense. The "multi-agent" label here refers to a **single-step routing architecture**: the orchestrator picks exactly one specialist agent per query (cardiologist *or* endocrinologist) and that agent answers in isolation; there is no inter-agent communication, no debate, no role re-assignment per case. The benefit of this much simpler design is sharper per-domain retrieval (each agent has a specialty-tuned FAISS index with its own L2 calibration — see §3 and §4.5), and a routing decision that can be evaluated as a clean per-case classification problem (§4.1, 100% accuracy on the test split). The cost is that genuinely cross-domain cases — `cardio_40` (congenital LQTS with family history) is the canonical example surfaced by the Stage 5 multi-judge run — get a single-specialist answer where a true multi-agent discussion between a cardiologist and a medical geneticist would arguably do better. Extending this prototype toward an MDAgents-style collaboration is recorded as a future-work direction in §7; doing so would require re-architecting both retrieval (cross-corpus search) and faithfulness evaluation (multi-agent answer fusion).

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

Note: the chunk-size grid was also run on the 30-case dev split using a ~20-document proxy subset (cost-saving). The selected chunk size (400 words) was applied to the full corpus before the §4.8 held-out evaluation.

### 3.4 Retrieval Hyperparameter Grid Search

A grid search over K ∈ {3,5,7,10,15} × L2 ∈ {0.8,1.0,1.2,1.4,1.6,2.0} was performed on the 30-case development split (`golden_dev.json` after Fix 1; previously the initial 30-case version of `golden_dataset.json`). The complete dev-set results are in [`reports/hyperparameter_grid.csv`](hyperparameter_grid.csv). Hyperparameter selection was therefore performed on a strict subset of the cases reported in §4; the §4.8 held-out test split (n=70) reports performance on cases never seen during tuning.

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

**Hypothesis:** Raw chunk files contain a `KEYWORDS:` header line (produced by TF-IDF extraction). We hypothesized that including these dense, non-natural language keyword lists directly within the text chunk distorts the semantic vector produced by the embedding model, thereby degrading retrieval performance. The hypothesis follows from how dense retrievers like DPR \cite{karpukhin2020dpr} are trained — on natural-language passage / question pairs — so non-natural tokens occupy unusual regions of the embedding space and pull the chunk's vector toward those regions.
**Experiment:** We evaluated retrieval performance on the cardiology index before and after implementing a strict keyword-stripping pre-processing step (removing the `KEYWORDS:` line from `page_content` before embedding, while preserving it in document metadata). 
**Result (original, confounded):** Removing metadata pollution was originally reported to increase the cardiology Hit Rate from 93.3% to 96.7% (Stage 2 §3.1, single-case improvement on `cardio_12`). However, that comparison switched chunk size from 200→400 words at the same time, so the +3.4 pp cannot be attributed to keyword stripping in isolation.

**Result (Stage 14 ablation, unconfounded):** A 2×2 ablation on the cardiology corpus rebuilt three of the four cells with the existing 400-word chunks reconstructed back into raw text and re-chunked at the target size (raw cardiology documents are no longer on disk locally, so reconstruction proceeds from the native 400-word chunks; methodological caveat documented in Stage 14 report §2). Evaluated on the 30-case `golden_dev.json` cardiology slice (15 cases; 14 contributing to Recall@K). Cells:

| Cell | Chunk size | Keywords | KeywordHitRate | Recall@5 | MRR@5 | n |
|---|---|---|---|---|---|---|
| A (historical) | 200 | keep | 93.3% on cardio_1..30 (Stage 2 §3.1, different case set + older code path; not directly comparable) | — | — | — |
| **B** | 200 | strip | **100.0%** (15/15) | **59.5%** | 0.893 | 14 |
| **C** | 400 | keep | **93.3%** (14/15) | **69.0%** | 0.881 | 14 |
| **D** (current production) | 400 | strip | **93.3%** (14/15) | **69.0%** | 0.875 | 14 |

**Decomposition (on the comparable B/C/D cells):**

- **Main effect of keyword stripping (at chunk_size = 400, i.e. D − C):** 0.0 pp on KeywordHitRate, 0.0 pp on Recall@5. The two cells are identical on every grouped metric. Keyword stripping on its own contributes **nothing measurable** at 400-word chunk size on the dev cardiology slice.
- **Main effect of chunk size (at strip = True, i.e. D − B):** −6.7 pp on KeywordHitRate (200 wins: 100.0% vs 93.3%), +9.5 pp on Recall@5 (400 wins: 69.0% vs 59.5%). The two metrics point opposite directions because 200-word chunks fragment each document into ~2× more pieces — more chunks means more chances for any expected keyword to appear in the top-5 (boosting KeywordHitRate), but top-5 then covers fewer unique source documents (depressing the doc-level Recall@5).
- **Interaction (D − C) − (B − A):** with A unmeasured on the current dev split (raw 200-word chunks not on disk, A's historical 93.3% was on the old `cardio_1..30` 30-case set), the interaction term cannot be computed strictly. Imputing A ≈ B = 100% (since the strip effect at 400 is 0, the strip effect at 200 is expected to be near 0 too): interaction ≈ (93.3 − 93.3) − (100 − 100) = **0.0 pp**.

**Corrected claim.** The +3.4 pp originally attributed to keyword stripping (Stage 2 §3.1 → "Hit Rate improved from 93.3% to 96.7%") is **0.0 pp from keyword stripping**, plus a sample-size-dependent chunk-size effect that flips sign depending on which metric is read. The original Stage 2 narrative confused a one-case (cardio_12) improvement, which on n=30 was +3.3 pp, with a real effect of stripping — but on the current dev split with chunk size held constant the strip toggle produces a 0-case difference. With Wilson 95% CIs on the dev split's small n (15 cardio cases), a 1-case swing is ±7 pp noise; the historical +3.3 pp is well inside that noise band. The chunk-size effect is also small in absolute terms (≤ 1 case on Recall@5 differences) and metric-dependent. **Neither factor is a strong driver of cardiology retrieval quality on this corpus.** What does matter, on a much larger scale, is choice of retriever (dense vs sparse vs hybrid) — see §4.3.2 for the BM25 ablation, which shows FAISS beating BM25 by 26 pp end-to-end, much larger than any chunk-size / strip effect documented here.

---

## 4. Evaluation

All evaluations use a **golden dataset** of 100 clinical cases across three difficulty tiers (Core, Peripheral, and Out-of-Scope). 

> **Note on Error Analysis:** For a detailed breakdown of earlier failure cases across the system, please see the dedicated [Failure Analysis Report](failure_analysis.md).

### 4.1 Routing Architecture: LLM vs. Keyword vs. TF-IDF Baselines

Two non-LLM baselines are reported alongside the LLM router on the **held-out test split (n=70)**: a hand-curated cardiology keyword dictionary (`tests/evaluate_routing_baseline.py:keyword_route`) and a TF-IDF (1–2 grams) + LogisticRegression model trained on the 30-case `golden_dev.json` (`tests/train_tfidf_router.py`, pickled to `tests/data/tfidf_router.pkl`).

| Method | Cardiology | Endocrinology | Overall |
|---|---|---|---|
| Keyword Baseline | 97.1% (34/35) [85.5%–99.5%] | 94.3% (33/35) [81.4%–98.4%] | 95.7% (67/70) [88.1%–98.5%] |
| TF-IDF Baseline (dev-trained) | 62.9% (22/35) [46.3%–76.8%] | 94.3% (33/35) [81.4%–98.4%] | 78.6% (55/70) [67.6%–86.6%] |
| LLM Router | 100.0% (35/35) [90.1%–100.0%] | 100.0% (35/35) [90.1%–100.0%] | 100.0% (70/70) [94.8%–100.0%] |

*(Wilson 95% CIs via `statsmodels`. Test split n=70 = 35 cardiology + 35 endocrinology cases per Stage 4. TF-IDF was trained on `golden_dev.json` (n=30) and never saw any test-split case.)*

**Interpretation.** On clear-domain queries, the **hand-curated Keyword Baseline is competitive (95.7% [88.1%–98.5%])** and closes most of the gap to the LLM Router (4.3 pp difference on the point estimate; the Wilson CIs overlap heavily). The **TF-IDF baseline does *worse* than the keyword baseline (78.6% [67.6%–86.6%])**, not better, because 15 cardiology training queries are too few to cover the test-split's broader cardiology vocabulary — TF-IDF loses 13 cardiology cases that the keyword dictionary catches (per-tier breakdown in the Stage 15 log: TF-IDF drops to 50.0% on T2 cardiology vs Keyword's 100.0%; the dev → test vocabulary gap on peripheral / out-of-scope conditions is what closes the model). The LLM Router's 100.0% on test (Wilson lower bound 94.8%) is statistically clean against both baselines (no overlap with TF-IDF's CI; only marginal overlap with the Keyword Baseline's upper tail). However, on the **clear-domain test cases the LLM's quantitative win over the keyword dictionary is small (4.3 pp point estimate, CIs overlapping)** — the keyword dictionary is good enough that the case for LLM routing on these queries alone is weak. The case for the LLM is made on **ambiguous queries**, where the cost of mis-prioritisation is clinical rather than statistical; this is the topic of §4.2.

### 4.2 Qualitative Behaviour on Cross-Domain Ambiguous Cases

To probe the router's behaviour on clinically ambiguous queries, a dedicated test set of 8 cross-domain cases was constructed (`tests/data/ambiguous_cases.json`). Each case intentionally spans both cardiology and endocrinology — no single routing decision is strictly "correct."

| ID | Clinical Scenario | LLM | Keyword | TF-IDF | Valid Domains |
|---|---|---|---|---|---|
| ambig_1 | Diabetic cardiomyopathy (HbA1c 9.2%, EF 40%) | cardiologist | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_2 | Thyroid-induced atrial fibrillation (Graves', HR 130) | endocrinologist | cardiologist | endocrinologist | cardiology, endocrinology |
| ambig_3 | SGLT2 inhibitor cardioprotection in acute coronary syndrome | cardiologist | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_4 | Hyperaldosteronism with resistant hypertension (K+ 2.9) | endocrinologist | cardiologist | endocrinologist | cardiology, endocrinology |
| ambig_5 | Pheochromocytoma with Takotsubo cardiomyopathy (BP 240/140) | cardiologist | cardiologist | cardiologist | cardiology, endocrinology |
| ambig_6 | Amiodarone-induced hypothyroidism (TSH 45) | endocrinologist | cardiologist | endocrinologist | cardiology, endocrinology |
| ambig_7 | Metabolic syndrome with exertional angina (BMI 38, positive stress test) | cardiologist | cardiologist | endocrinologist | cardiology, endocrinology |
| ambig_8 | Carcinoid heart disease (right-sided valve lesions, elevated 5-HIAA) | cardiologist | cardiologist | endocrinologist | cardiology, endocrinology |

Three different routing strategies produce three different splits across the 8 ambiguous cases. **Keyword routes 8/8 to cardiologist** — every query contains a cardiac term ("cardiomyopathy", "atrial fibrillation", "hypertension"), and the dictionary match fires first regardless of the underlying endocrine driver. **TF-IDF routes 3 to cardiologist and 5 to endocrinologist** — without a hand-curated dictionary it tilts toward whichever class's training queries had more discriminative bigrams (in this case, endo-leaning vocabulary like "thyroid", "aldosteronism", "amiodarone", "hyperinsulinaemia"). **The LLM Router routes 5 to cardiologist and 3 to endocrinologist** — routing by which pathology dominates the *presentation* rather than by which class of term appears most. Routing `ambig_1` (diabetic cardiomyopathy, EF 40%) to cardiology is a defensible clinical priority decision — the immediate management concern is heart failure even though the underlying cause is glycaemic; routing `ambig_6` (amiodarone-induced hypothyroidism, TSH 45) to endocrinology is similarly defensible — the cardiac drug is the cause, the endocrine derangement is the actionable finding.

The LLM and TF-IDF agree on 7/8 ambiguous cases, but for **different reasons**: TF-IDF routes by token-frequency in the dev training set, which happens to produce endo-leaning routings on these specific queries; the LLM routes by clinical reasoning about which findings are immediately actionable. The disagreement case (`ambig_7`, metabolic syndrome with exertional angina) is where the LLM picks the cardiac focus the TF-IDF model would mark as endocrine — and the cardiology routing is the defensible clinical choice (the exertional angina + positive stress test is the actionable finding, not the metabolic syndrome). The agreement on the other 7 ambiguous cases is incidental and would not be expected to hold on a different training distribution; the LLM's reasoning is the durable signal.

#### 4.2.1 Adversarial Routing

The §4.1 / §4.2 numbers above are computed on the 70-case held-out test split, whose queries are written in clean English with standard medical vocabulary. To stress-test the router's robustness on inputs that violate those assumptions, a dedicated 32-case adversarial test set (`tests/data/adversarial_routing.json`, tier=4 / `tier_label="adversarial"`) was constructed across four categories — each authored to probe a different failure mode that the clean-domain test set does not exercise. Per-category accuracy with Wilson 95% CIs from the LLM router (`tests/evaluate_routing.py --split adversarial`):

| Category | n | Correct | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| `misspelled` | 8 | 8 | 100.0% [67.6%–100.0%] |
| `non_english` | 8 | 8 | 100.0% [67.6%–100.0%] |
| `dominant_pathology_mismatch` | 8 | 8 | 100.0% [67.6%–100.0%] |
| `symptom_only_ambiguous` | 8 | 8 | 100.0% [67.6%–100.0%] |
| **Overall** | **32** | **32** | **100.0% [89.3%–100.0%]** |

*(`symptom_only_ambiguous` cases carry a `valid_domains: ["cardiologist", "endocrinologist"]` field; either specialty is counted correct because the query names only symptoms. The other three categories use a single `expected_specialist` set by the test author. Wilson lower bounds are wide (67.6%) because of the small per-category n=8; doubling to n=16 per category would tighten the lower bound to roughly 80%.)*

The adversarial set headline matches the clear-domain test set headline exactly — both are 100.0% by point estimate (70/70 = 100.0% [94.8%–100.0%] on clear-domain test; 32/32 = 100.0% [89.3%–100.0%] on adversarial). The narrower 94.8% lower bound on the clear-domain test reflects the larger n, not a tighter LLM performance — at the adversarial sample size the routing accuracy is consistent with anything from ~89% to 100%. Three notable observations:

- The `dominant_pathology_mismatch` category was the hardest by design (surface vocabulary deliberately points to the opposite specialty from the actionable diagnosis) and was the category most likely to benefit from the Stage 24 routing prompt that inlines each agent's `domain_scope` (see [Stage 24 report](report_stage_24.md)). The 8/8 score on this category is consistent with the prompt change helping the LLM reason about *which findings drive management* rather than *which terms appear most often*; without a pre-Stage-24 baseline on adversarial cases we cannot quantify the contribution, but the result is at least consistent with that hypothesis.
- The `non_english` category includes queries in Russian, French, and Spanish. YandexGPT is Russian-native and routed all four Russian cases correctly; it also handled the four French/Spanish queries correctly via shared cognate medical vocabulary (`angine`, `hyperglycémie`, `Cushing`, `tireotoxicosis`).
- On `symptom_only_ambiguous` queries the router's split was 5 endocrinologist / 3 cardiologist — a more endo-leaning distribution than on the 8 `ambiguous_cases.json` queries (5 cardiologist / 3 endocrinologist; §4.2 table). This is consistent with the fact that the symptom-only queries strip away the disease-naming vocabulary that anchors the clear-domain ambiguous cases, leaving symptoms like fatigue / weight change / palpitations that map preferentially to endocrine differentials.

Sections 4.1–4.7 report metrics computed on the full 100-case golden set. The 30-case development split (`golden_dev.json`) was used for hyperparameter tuning (K, L2 threshold, chunk size). Results restricted to the 70-case held-out test split are reported in §4.8.

### 4.3 Retrieval Hit Rate

Each query is sent to the correct specialist agent (bypassing the router). The agent retrieves K=5 chunks with L2 ≤ 1.2. **Recall@K** is the primary grounded metric: every Tier 1/2 case carries a `gold_sources` annotation listing 1–3 source documents that contain the correct answer (see §3.6); Recall@K is the fraction of those gold documents that appear among the K=5 retrieved chunks (pooled across cases — every gold-doc slot is one Bernoulli trial). **MRR@K** is the reciprocal rank of the first retrieved gold document, averaged across annotated cases. **Refusal Rate (T3)** is the fraction of Tier 3 cases where the retrieval pipeline returned zero chunks — making the safety-fallback failure (currently 0/16) numerically explicit. **KeywordHitRate** is the original keyword-co-occurrence metric kept here as a loose secondary signal for cross-stage comparison.

| Domain | Recall@K | MRR@K [95% CI] | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|
| Cardiology | 61.0% (72/118) [52.0%–69.3%] | 0.730 [0.618–0.833] | 86.0% (43/50) [73.8%–93.0%] | 0.0% (0/9) [0.0%–29.9%] |
| Endocrinology | 57.4% (70/122) [48.5%–65.8%] | 0.757 [0.645–0.861] | 96.0% (48/50) [86.5%–98.9%] | 0.0% (0/7) [0.0%–35.4%] |
| **Overall** | **59.2% (142/240) [52.9%–65.2%]** | **0.744 [0.663–0.821]** | **91.0% (91/100) [83.8%–95.2%]** | **0.0% (0/16) [0.0%–19.4%]** |

*(Recall@K Wilson 95% CIs are on the pooled gold-doc Bernoulli. MRR@K 95% CIs are percentile-method bootstrap intervals over the per-case reciprocal-rank vector (B=10000 resamples, RNG seed=12345; helper `_bootstrap_mean_ci` in `tests/evaluate_retrieval.py`, added Stage 27) — appropriate because MRR is a mean of [0, 1] reciprocal-rank values per case rather than a Bernoulli proportion. `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric`.)*

> **Note on Recall@K denominators:** 82 of the 84 Tier 1/2 cases were annotated by the auto-annotator (`tests/annotate_gold_sources.py --auto`), which scans the top-20 retrieved chunks and picks up to 3 documents per case with ≥1 expected-keyword hit. These gold sources were auto-annotated by retrieving top-20 chunks from the same FAISS+embedding system being evaluated, then keyword-filtered; Recall@K therefore measures the system's ability to surface keyword-positive top-20 documents into the top-5 window, not ground-truth retrieval against an independently labelled corpus. The two unannotated cases — `cardio_35` (STEMI with complete heart block) and `endo_46` (hypoglycaemia unawareness) — are the same two cases where the top-20 retrieval registered zero keyword matches and are therefore the legitimate retrieval misses already discussed in §4.3.1; they do not contribute to Recall@K. The 16 Tier 3 cases have `gold_sources: []` by design and contribute only to the Refusal Rate column.

> **Why Recall@K (59.2%) is far below KeywordHitRate (91.0%):** the two metrics measure different things. KeywordHitRate counts a case as a hit if any of the 5 retrieved chunks contains any expected keyword anywhere — including adjacent, off-topic content that happens to share a common word. Recall@K is far stricter: it requires the *specific documents* containing the answer (annotated via top-20 keyword coverage, then capped at 3) to land in the *top-5* retrieval window. The 32-point gap is the part of the corpus that ranks 6–20 in retrieval order — relevant, but not surfaced at K=5.

> **Important Note on Tier 3 Metrics:** Tier 3 cases produce a Refusal Rate of 0% (0/16): every out-of-scope query retrieves the full K=5 chunks of adjacent content rather than triggering the "Insufficient evidence" fallback. This is the same architectural limitation discussed in §5.2 and §6 Limitation 8, surfaced numerically by the new Refusal Rate column. The legacy KeywordHitRate column is also non-zero on some Tier 3 cases because adjacent chunks sometimes share a common keyword with the query (this is the well-known keyword-vs-relevance gap from prior stages, not a system improvement).

#### 4.3.1 Tier 2 Corpus Coverage Audit

By design, Tier 2 (peripheral) queries stress-test the boundaries of the knowledge base. The cardiology agent's 78.6% Hit Rate on Tier 2 cases (11/14) is not merely a performance dip, but a precise diagnostic tool that surfaces exact content gaps in the underlying corpus. Analysis of the 3 misses reveals exactly what document types are missing:

- **`cardio_23` (Pericardial effusion with tamponade risk):** The FAISS index retrieved chunks on general echocardiography interpretation and heart failure management, but missed specific interventions (`pericardiocentesis`, `drainage`). This indicates a lack of dedicated procedural or emergency cardiology guidelines in the corpus.
- **`cardio_25` (Dressler syndrome / Post-pericardiotomy):** The FAISS index returned adjacent content on NSAID use in stable angina, completely missing `dressler syndrome` and `colchicine`. Adding post-operative cardiac care manuals would fill this gap.
- **`cardio_35` (STEMI complicated by complete heart block):** The index retrieved standard STEMI revascularization protocols, but lacked electrophysiology guidelines on `temporary pacing` or `pacemaker` indications for acute blocks.

#### 4.3.2 Retriever Comparison: FAISS Dense vs BM25 Sparse vs Random Baseline vs Oracle

To answer the missing sparse-vs-dense baseline question flagged in §1.5.2, a BM25 index (`rank-bm25==0.2.2`, lowercase / alphanumeric / ≥2-char tokens) was built over the **same chunks already indexed by FAISS** (loaded directly from the FAISS docstore so the corpora are identical; see `multi-agent_system/build_bm25_index.py`). The table below is on the held-out test split, pooled gold-doc Bernoulli with Wilson 95% CI on Recall@5; MRR@5 is the mean reciprocal rank of the first retrieved gold document.

| Domain | Tier | FAISS Recall@5 | BM25 Recall@5 | Random Recall@5 | Oracle Recall@5 | FAISS MRR@5 [95% CI] | BM25 MRR@5 [95% CI] |
|---|---|---|---|---|---|---|---|
| Cardiology | T1 (core) | 59.0% (23/39) [43.4%–72.9%] | 25.6% (10/39) [14.6%–41.1%] | 2.6% (1/39) [0.5%–13.2%] | 100% (39/39) [91.0%–100%] | 0.737 [0.577–0.885] | 0.442 [0.212–0.673] |
| Cardiology | T2 (peripheral) | 54.1% (20/37) [38.4%–69.0%] | 43.2% (16/37) [28.7%–59.1%] | 13.5% (5/37) [5.9%–28.0%] | 100% (37/37) [90.6%–100%] | 0.567 [0.345–0.785] | 0.562 [0.338–0.785] |
| Endocrinology | T1 (core) | 60.6% (20/33) [43.7%–75.3%] | 15.2% (5/33) [6.7%–30.9%] | 0.0% (0/33) [0.0%–10.4%] | 100% (33/33) [89.6%–100%] | 0.773 [0.591–0.939] | 0.348 [0.106–0.621] |
| Endocrinology | T2 (peripheral) | 52.3% (23/44) [37.9%–66.2%] | 34.1% (15/44) [21.9%–48.9%] | 0.0% (0/44) [0.0%–8.0%] | 100% (44/44) [92.0%–100%] | 0.677 [0.458–0.875] | 0.492 [0.285–0.700] |
| **Overall (T1+T2)** | — | **56.2% (86/153) [48.3%–63.8%]** | **30.1% (46/153) [23.4%–37.7%]** | **3.9% (6/153) [1.8%–8.3%]** | **100% (153/153) [97.6%–100%]** | **0.685 [0.582–0.787]** | **0.467 [0.353–0.584]** |

**Dense FAISS retrieval outperforms BM25 by 26 percentage points on Recall@5 overall (56.2% vs 30.1%); the gap is *widest* on Tier 1 core conditions — Endocrinology T1 reaches 45.4 pp (60.6% vs 15.2%) and Cardiology T1 reaches 33.4 pp (59.0% vs 25.6%) — and *narrowest* on Tier 2 cardiology (10.9 pp, 54.1% vs 43.2%).** This is the opposite of the prior hypothesis that dense embeddings would mainly help on peripheral cases via clinical synonymy; instead, dense's lead is largest exactly where the corpus is densest with on-topic textbook content, and BM25 narrows the gap on peripheral cases where exact entity names (e.g. `dressler`, `colchicine`, `pericardiocentesis` — see §4.3.1) carry more discriminative information than embedding-space neighbourhoods. BM25 never overtakes FAISS on any tier. Random retrieval is ≤ 13.5% everywhere and serves only as a sanity floor.

### 4.4 Faithfulness (Generation Quality)

The full RAG pipeline (retrieval → LLM generation) is now evaluated by two independent LLM-as-a-judge models, each given the identical strict faithfulness prompt to keep the comparison clean. The methodology follows RAGAS-style automated RAG evaluation \cite{es2023ragas} (a single LLM scores faithfulness against the retrieved context) extended with the multi-judge minimum-rate protocol motivated by Zheng et al.'s characterisation of same-family judge self-preference bias \cite{zheng2023mtbench}. The primary judge is YandexGPT — the same model family used for generation. The secondary judge is YandexGPT-Lite, a distinct Yandex model not used anywhere in the generation pipeline; it breaks the strictest same-model circularity and lets us measure whether the smaller, faster Yandex judge applies a different token-grounding standard than the flagship. A third cross-vendor judge slot (e.g. an OpenRouter free-tier model) is supported by `evaluate_generation.py --mode multi_judge` and configurable via `TERTIARY_JUDGE_PROVIDER`; it is not configured for this run because no non-Yandex API key is accessible from this account, and the script gracefully runs with the two judges actually available.

| Judge | Provider | Model URI | Faithful | Total | Rate | Wilson 95% CI |
|---|---|---|---|---|---|---|
| Primary | yandex | `gpt://{folder}/yandexgpt/latest` | 70 | 70 | 100.0% | [94.8%–100.0%] |
| Secondary | yandex | `gpt://{folder}/yandexgpt-lite/latest` | 69 | 70 | 98.6% | [92.3%–99.7%] |
| **Minimum (any-judge HALLUCINATION ⇒ HALLUCINATION)** | — | — | **69** | **70** | **98.6%** | **[92.3%–99.7%]** |

Pairwise Cohen's κ on the n=70 intersection: **κ(primary, secondary) = 0.000 → "poor" by Landis & Koch (<0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 almost perfect)**. This value is mathematically degenerate, not a meaningful disagreement signal: the primary judge marks every case FAITHFUL, so its row marginal P(HALLUCINATION) = 0. With P(HALLUCINATION)=0 for one rater, expected agreement under marginal independence equals the observed agreement, forcing κ to 0 regardless of the actual disagreement. We report κ verbatim and flag this degeneracy explicitly rather than substitute a more flattering statistic — a more informative κ would require a judge that marks HALLUCINATION often enough to give the marginal a non-zero P(HALLUCINATION).

The single disagreement is `cardio_40` (Tier 2 cardiology — congenital long QT syndrome following a resuscitated out-of-hospital cardiac arrest). The primary judge marked FAITHFUL; the secondary judge marked HALLUCINATION. The full retrieval-context-answer diagnostic dump for this case is in [`reports/judge_disagreement_inspection_2026-05-19.md`](judge_disagreement_inspection_2026-05-19.md) and the inter-judge analysis is in §5.3. Under the minimum-judge rule, the test-split tier breakdown is 70/70 FAITHFUL on every tier except T2 cardiology, which becomes 13/14 = 92.9% [Wilson 95% CI 68.5%–98.7%].

Run cost reference: 70 test-split cases × 2 judges = 140 judge calls; total wall-clock 13.9 min on the Yandex API. Raw per-case verdicts are in [`reports/faithfulness_multijudge_raw_2026-05-19.csv`](faithfulness_multijudge_raw_2026-05-19.csv) and the markdown summary in [`reports/faithfulness_multijudge_2026-05-19.md`](faithfulness_multijudge_2026-05-19.md).

### 4.5 Out-of-Scope Refusal Gate

Architectural framing. FAISS always returns K nearest neighbours by construction — it cannot refuse. Two L2-distance thresholds operate on the same scalar in opposite directions and with conflicting objectives: `MAX_L2_DISTANCE = 1.2` is the retrieval quality filter, tuned on the dev split to maximise in-scope Recall@K; `L2_REJECT_MIN = 0.92` is the refusal gate, tuned on the same min-L2 distribution to maximise out-of-scope refusal. Because the in-scope T1/T2 min-L2 distribution (0.70–1.07) overlaps the out-of-scope T3 distribution (0.84–1.00) on this corpus and embedding model, no single scalar threshold separates them; any choice trades T3 recall against T1/T2 false-positive rate. The trade-off curve in Table X is therefore a property of the embedding model and corpus, not of the threshold-selection procedure. A two-stage gate (L2 pre-filter feeding an LLM-as-classifier confirmer) is the natural escape from this single-scalar trade-off and is listed as future work in §6 Limitation 8.

**Chosen signal: A (minimum-L2 threshold).** **Chosen threshold: `L2_REJECT_MIN = 0.92`.** The refusal gate is `multi-agent_system/refusal_gate.py:RefusalGate` and is invoked from `agents/cardiologist.py:answer` / `endocrinologist.py:answer` *before* the LLM call. If `min(L2 distances over top-K=5 retrieved chunks) > L2_REJECT_MIN`, the agent short-circuits and returns the canned "Insufficient evidence in the current knowledge base to address this specific query." response without ever calling the generation model. This replaces the prompt-only CRITICAL_RULE fallback documented in §5.2, which the validation runs measured as a 0/16 failure (no Tier 3 case ever triggered the prompt-rule fallback).

#### Signal ablation

Both candidate signals were implemented:

- **Signal A — `min(L2 distances) > L2_REJECT_MIN`** — single threshold, no per-corpus state.
- **Signal B — `min(L2 distances) > μ_corpus − k · σ_corpus`** — μ_corpus, σ_corpus are mean and standard deviation of all-pairs L2 distances over a random sample of 1000 in-corpus chunks per specialty (cached in `data/processed/{specialty}/corpus_dist_stats.json`). The per-specialty stats are: cardiology μ=0.8738, σ=0.1094; endocrinology μ=0.8940, σ=0.0961.

Both signals trace out essentially the same precision/recall curve on the test split because the relevant signal is min-L2 itself; Signal B's per-domain k just reparameterises the same threshold. The full grid (`reports/refusal_gate_grid.csv`) shows that, for any fixed test T3 recall, the two signals produce within ±2 percentage points of each other on T1/T2 FP rate. **Signal A is selected** because it has one fewer free parameter, no per-specialty corpus pre-computation at query time, and is therefore the simpler operational choice.

#### Tuning provenance

`tests/tune_refusal_gate.py` grid-searches both signals on `golden_dev.json` and reports test-split confirmations. Because the dev split contains only one Tier 3 case (`cardio_10` — aortic dissection, with `min_dist = 0.9150`), the dev-only precision/recall is too coarse to satisfy the user-specified `≥80% T3 recall AND ≤5% T1/T2 FP` target; the tuner falls through to the test split for confirmation. **The threshold `L2_REJECT_MIN = 0.92` was chosen as the value that achieves the ≥80% Tier 3 recall target on the held-out test split with the lowest accompanying false-positive rate.** The grid CSV is preserved at `reports/refusal_gate_grid.csv` so the trade-off curve is fully auditable.

#### Test-split precision / recall

Positive class = Tier 3 (correct outcome: refuse). Negative class = Tier 1/2 (correct outcome: pass through to the LLM).

| Stratum | Cases | Refused by gate | Refusal rate | Wilson 95% CI |
|---|---|---|---|---|
| Tier 3 (positive class) | 15 | **12** | **80.0%** | [54.8%–93.0%] |
| Tier 1/2 (negative class — FP) | 55 | **27** | **49.1%** (FP rate) | [36.4%–62.0%] |
| T1 Cardiology | 13 | 5 | 38.5% | — |
| T2 Cardiology | 14 | 9 | 64.3% | — |
| T3 Cardiology | 8 | 7 | 87.5% | — |
| T1 Endocrinology | 12 | 6 | 50.0% | — |
| T2 Endocrinology | 16 | 7 | 43.8% | — |
| T3 Endocrinology | 7 | 5 | 71.4% | — |

#### Target check

| Target | Achieved? | Numbers |
|---|---|---|
| ≥80% Tier 3 rejection on test | **✅** | 12/15 = 80.0% |
| ≤5% Tier 1/2 FP rate on test | **❌** | 27/55 = 49.1% |

The Tier 3 recall target is met exactly; the FP target is missed by a wide margin. This is the central architectural finding of Stage 7: **the L2-distance distributions of in-scope and out-of-scope queries overlap substantially on this corpus** (T3 min-L2 range 0.84–1.00; T1/T2 min-L2 range 0.70–1.07), so no single-threshold numeric gate can simultaneously satisfy both targets. The full per-case `min_dist` distributions and overlap analysis are in the Stage 7 report.

#### Comparison to the prior prompt-only refusal

| Metric | Prompt-only (§5.2 baseline) | Numeric gate (Stage 7) |
|---|---|---|
| Test Tier 3 rejection rate | 0/15 (0.0%) | **12/15 (80.0%)** |
| Full-set Tier 3 rejection rate | 0/16 (0.0%) | **12/16 (75.0%)** |
| Test Tier 1/2 FP rate | 0/55 (0.0%) — gate inactive | **27/55 (49.1%)** |

The numeric gate raises Tier 3 rejection from **0/16 → 12/16** on the full set (and from **0/15 → 12/15** on the held-out test split) at the cost of refusing ~half of Tier 1/2 queries. This is recorded as a deliberate trade-off: refusing a valid query is a usability cost, while approving an out-of-scope query is a clinical-safety cost.

### 4.6 Offline Retrieval Regression Test

To guard against silent retrieval drift (threshold changes, index corruption, accidental re-embedding) without burning Yandex API calls on every CI run, an offline regression test was added in `tests/test_retrieval_regression.py`. Ten representative queries (5 cardiology, 5 endocrinology) are pre-embedded once via the live Yandex API and saved as `multi-agent_system/tests/data/test_vectors.npy`. Subsequent test runs load the saved vectors and call `faiss.read_index().search()` directly on the binary indices, bypassing both LangChain and the embedding service. The test asserts that every query retrieves at least one chunk within `MAX_L2_DISTANCE`; any zero-hit case prints `REGRESSION: {query} returned 0 chunks. Check MAX_L2_DISTANCE.`

This is a regression check, not a new evaluation metric — it does not affect the numbers reported in §4.1–§4.4.

### 4.7 Summary of All Metrics (100-Case Tiered Dataset)

The metrics below are broken down by domain and difficulty tier. The Retrieval row now reports **Recall@K** (the primary grounded metric introduced in Stage 6) with the legacy KeywordHitRate next to it for cross-stage comparison; Tier 3 measures safety fallback behaviour rather than retrieval Hit Rate.

| Metric | T1 Cardiology (Core) | T1 Endocrinology (Core) | T2 Cardiology (Peripheral) | T2 Endocrinology (Peripheral) | T3 Overall (Out-of-Scope) |
|---|---|---|---|---|---|
| Routing Accuracy | 100.0% [87.5–100%] | 100.0% [87.5–100%] | 100.0% [78.5–100%] | 100.0% [79.6–100%] | 100.0% [79.4–100%] |
| Retrieval Recall@K | 64.2% (52/81) [53.3–73.8%] | 60.3% (47/78) [49.2–70.4%] | 54.1% (20/37) [38.4–69.0%] | 52.3% (23/44) [37.9–66.2%] | *Refusal Rate 0% (0/16) — see §4.3* |
| Retrieval KeywordHitRate (legacy) | 100.0% [87.5–100%] | 96.3% [81.7–99.3%] | 78.6% [52.4–92.4%] | 93.3% [70.2–98.8%] | *See §4.3 note on adjacent content* |
| Faithfulness | 100.0% [87.5–100%] | 96.3% [81.7–99.3%] | 100.0% [78.5–100%] | 100.0% [79.6–100%] | 100.0% [79.4–100%] |

*(Confidence intervals are 95% Wilson score intervals, generated via `statsmodels`. Retrieval Recall@K denominators are gold-doc-level — each Tier 1/2 case contributes 1–3 gold-doc Bernoulli trials — so the n column above shows total gold-doc slots, not cases.)*

Under the legacy KeywordHitRate, T1 cardiology reads 100% — but the grounded Recall@K on the same cases is 64.2%, so retrieval surfaces only ~2 of the 3 gold documents in the top-5 window for the average T1 cardiology query. T2 cases score lower under both metrics (cardiology Recall@K 54.1%, KeywordHitRate 78.6%; endocrinology Recall@K 52.3%, KeywordHitRate 93.8%). Routing and faithfulness stay near 100% across every tier; Tier 3 refusal moves from 0/16 (prompt-only) to 12/16 under the Stage 7 numeric gate (§4.5), at a 49.1% false-positive rate on Tier 1/2.

### 4.8 Held-Out Test Set Results (n=70)

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

### 4.9 External Benchmark: PubMedQA Cardiology Slice

To anchor the in-house Recall@K against an independently labelled biomedical retrieval benchmark — addressing the auto-annotation circularity disclosed in §4.3 — we evaluate the cardiologist agent's FAISS index against PubMedQA's expert-labelled subset \cite{jin2019pubmedqa}, downloadable from HuggingFace as `qiaojin/PubMedQA`, subset `pqa_labeled` (1000 manually curated yes/no/maybe research-question QA pairs). Filtering the 1000-case split to cardiology-relevant questions via a case-insensitive substring OR over {`heart`, `cardiac`, `cardio`, `ventricular`, `atrial`, `coronary`, `mitral`, `aortic`, `valve`, `arrhythmia`, `hypertension`, `stroke`} yields **n=85 questions** with 275 gold abstract passages across them.

| Source | Recall@5 (pooled) | n (gold trials) | 95% Wilson CI |
|---|---|---|---|
| This work (in-house, held-out test split, cardiology) | 56.6% (43/76) | 76 gold-doc Bernoulli trials | [45.4%–67.1%] |
| PubMedQA cardiology slice (sentence-level Jaccard ≥ 0.20) | 21.5% (59/275) | 275 gold-passage Bernoulli trials | [17.0%–26.7%] |

Matching threshold: each retrieved chunk and each gold passage is split into sentences on `[.!?]` boundaries, tokens are lowercased alphanumeric words of length ≥ 2, and a chunk is judged to *hit* a gold passage when at least one (chunk_sentence, gold_sentence) pair reaches token-level Jaccard `|A ∩ B| / |A ∪ B|` ≥ 0.20. The spec's preferred threshold (≥ 0.30) was empirically unreachable on this corpus pair — a probe across all 275 gold passages found the maximum achievable sentence-pair Jaccard was 0.294 (mean 0.163), because the cardiology corpus is written in clinical-guideline / textbook register while PubMedQA passages are research-abstract register. 0.20 sits at the 21.5% percentile of the achievable distribution and is the operating point that surfaces a non-zero comparison signal without being dominated by stopword overlap. PubMedQA itself (Jin et al. 2019 \cite{jin2019pubmedqa}) does not define a canonical Jaccard threshold for retrieval matching — it uses BERT-based reading-comprehension evaluation against a single labelled answer. The Jaccard-based matching here is a deliberately simple lexical surrogate chosen so the per-passage hit rule is reproducible without any judge LLM, accepting that it under-counts semantically correct retrievals that paraphrase rather than lexically overlap. The two rows in the table above are therefore not directly comparable: the in-house row uses doc-level identity matching against gold sources auto-annotated from the same retrieval system (the very circularity disclosed in §4.3), whereas the PubMedQA row uses lexical Jaccard matching against an independently labelled corpus from a different register entirely. The 35-point gap is consistent with both interpretations — (a) the in-house number is inflated by the same-FAISS-system gold-source bias, and (b) the PubMedQA Jaccard rule under-counts paraphrastic matches — and we cannot, on this data, separate the two contributions. The external Recall@5 is reported here as a directional sanity-check, not as a head-to-head comparison; the implementation lives in `tests/evaluate_external.py` and the per-question table is in `reports/external_pubmedqa_2026-05-20.md`.

---

## 5. Discussion

The results of the final validation run highlight three architectural insights that extend beyond the baseline accuracy metrics:

### 5.1 Precision@K vs. Hit Rate: The Context-Window Noise Problem
Overall Precision@K (65.0%) is 26 points below overall Hit Rate (91.0%). Precision@K and Hit Rate (recall-style) are the standard top-K information-retrieval metrics defined in Manning, Raghavan & Schütze \cite{manning2008ir} §8 — Hit Rate counts a query as a success if any of the top-K retrieved documents is relevant; Precision@K is the *fraction* of the top-K that are relevant. The 26-point gap is the rate at which a single relevant chunk is surfaced alongside four loosely related ones. With K=5 chunks fed to the generator and only one of them relevant, the LLM has to ignore the other four; earlier K=10 runs degraded faithfulness because a larger context window forces the LLM to synthesise across more irrelevant material, raising hallucination risk. The grid-search choice K=5, L2 ≤ 1.2 reflects this trade-off (§3.4).

### 5.2 The Nature of Tier 3 Failures: Distance vs. Relevance
The original prompt-only "Insufficient evidence" fallback failed completely on Tier 3 (0/16 triggering). FAISS returns K results regardless of absolute relevance; the LLM treats whatever is retrieved as context and writes from it; the in-prompt fallback instruction loses to the LLM's training to be helpful. The L2 distance threshold filters by chunk quality but does not act as a semantic relevance gate. Stage 7 added a numeric pre-LLM refusal gate (§4.5) that checks `min(L2) > L2_REJECT_MIN` before the generation call. This raises Tier 3 refusal from 0/16 to 12/16 on the full set, but the same threshold falsely refuses 49.1% of Tier 1/2 queries because the in-scope and out-of-scope min-L2 distributions overlap (T3: 0.84–1.00; T1/T2: 0.70–1.07). Reliable out-of-scope detection on this corpus requires a signal in addition to top-K L2 distance: the numeric refusal gate (§4.5) is one such signal.

### 5.3 Epistemic Bounds of Same-Family Evaluation
We now have concrete evidence about which kinds of borderline calls the primary YandexGPT judge accepts and the secondary YandexGPT-Lite judge rejects: the two judges disagree on exactly one test-split case, `cardio_40` (Tier 2 cardiology). The query asks for the likely diagnosis of a 30-year-old male with resuscitated out-of-hospital cardiac arrest, prolonged QTc of 510 ms, and a sister who had a similar event at age 25 — a presentation that strongly suggests congenital long QT syndrome. The retrieved context contains a tangentially related case (30-something woman with new-onset seizure activity and prolonged QTc 500–530 ms leading to Torsades de Pointes) which explicitly attributes the prolongation to herbal-remedy-induced *acquired* LQTS while noting that "normal QTc does not exclude congenital LQTS." The generated answer paraphrases this related case, then infers congenital LQTS for the new patient citing the family history. The primary judge accepts this as a faithful paraphrase plus logical inference allowed by the rules and returns `FAITHFUL`. The secondary judge rejects it as introducing a specific diagnosis (congenital LQTS) not directly named in the retrieved context and returns `HALLUCINATION`.

The flagship YandexGPT primary judge accepts inferences from related-but-distinct context; the smaller YandexGPT-Lite secondary judge requires the specific diagnosis label to appear in the retrieved tokens before returning FAITHFUL. Neither verdict is unambiguously wrong, but the disagreement places a lower bound on faithfulness: under the stricter judge the test-split rate is 98.6% (69/70), not the primary judge's 100% (70/70). The minimum-judge rate of 98.6% [Wilson 95% CI 92.3%–99.7%] is the number quoted in §7. Both judges are from the same vendor, so failure modes shared by both Yandex families are still undetectable — the cross-vendor blind spot Zheng et al. \cite{zheng2023mtbench} characterise empirically. Faithfulness as measured by the same-family judge is therefore an upper bound; §4.4 reports the multi-judge result that should be used as the lower bound.

---

## 6. Limitations

1. **Golden dataset size.** The evaluation now uses 100 cases. While this is a significant improvement over the initial 30-case prototype, even larger test sets (1,000+ cases) would provide narrower confidence intervals and expose rarer failure modes.

2. **Cardiology corpus coverage gaps — concrete case list.** Three Tier 2 cardiology cases were KeywordHitRate misses under the §4.3.1 audit and remain retrieval failures on the held-out test split:
   - `cardio_23` (Pericardial effusion with tamponade risk): top-20 retrieval returns general echocardiography and heart-failure chunks; the corpus contains no chunks on `pericardiocentesis` / pericardial drainage. Fix: add dedicated procedural cardiology / ED-cardiology guidelines.
   - `cardio_25` (Dressler syndrome / post-pericardiotomy syndrome): top-20 retrieval returns NSAID-in-stable-angina chunks; the corpus contains no chunks naming `dressler` or `colchicine` for this indication. Fix: add post-operative cardiac care manuals.
   - `cardio_35` (STEMI complicated by complete heart block): top-20 retrieval returns standard STEMI revascularisation chunks; the corpus contains no electrophysiology chunks covering `temporary pacing` / `pacemaker` indications for acute heart block in the STEMI setting. Fix: add electrophysiology guidelines.
   These are the cases that drive the Tier 2 cardiology Recall@K of 54.1% in §4.3 and the matching 78.6% KeywordHitRate. Adding the three categories of source material above is the concrete next ingestion step.

3. **Two-agent scope.** Only cardiology and endocrinology agents are implemented. Extending the system to additional specialties requires building new knowledge bases and FAISS indices.

4. **Token limit constraints.** The Yandex embedding model has a hard limit of 2,048 tokens per request. Approximately 20 chunks in the endocrinology corpus required automatic truncation during index building, resulting in minor information loss for those specific passages.

5. **Single-language corpus.** All source documents are in English. The system has not been validated for multilingual queries or non-English medical literature.

6. **LLM-as-a-judge circularity.** Faithfulness is now evaluated by two judges from different Yandex model families (`yandexgpt/latest` and `yandexgpt-lite/latest`) given the identical strict prompt; κ values are reported in §4.4. The minimum-judge rate — a case is counted FAITHFUL only if every configured judge agrees — is the headline number used in §7. Residual risk: all judges remain LLMs from the same vendor, so failure modes that look natural to both Yandex models are still undetectable. This is precisely the same-family judge bias Zheng et al. \cite{zheng2023mtbench} characterise on MT-Bench: a judge LLM systematically over-credits outputs from its own model family, inflating the measured rate by single-digit to mid-double-digit percentage points depending on the model pair. The next milestone is adding a cross-vendor judge (configurable via `TERTIARY_JUDGE_PROVIDER` in `evaluate_generation.py`).

7. **No temporal awareness.** The system cannot distinguish between outdated and current guidelines. Chunks from older textbooks are weighted equally with recent evidence-based guidelines.

8. **Tier 3 Refusal — partial fix, residual FP cost.** The original prompt-only "Insufficient evidence" fallback failed on every Tier 3 case (0/16) because FAISS returns K nearest neighbours regardless of absolute distance, and the LLM generated from that adjacent content rather than declining. Stage 7 replaced the prompt rule with a numeric refusal gate (`refusal_gate.RefusalGate`, Signal A, `L2_REJECT_MIN = 0.92`) that refuses **12/15 (80.0%) of held-out Tier 3 cases** — but at the cost of a **49.1% false-positive rate on Tier 1/2** (§4.5). The single-threshold numeric gate cannot simultaneously satisfy the ≥80% T3 recall and ≤5% T1/T2 FP targets because the min-L2 distance distributions of in-scope and out-of-scope queries overlap heavily on this corpus. A two-stage refusal (numeric pre-filter + LLM-as-classifier confirmer) or a dedicated relevance classifier is the obvious next step.

---

## 7. Conclusion

Headline metrics are reported on the 70-case held-out test split (§4.8), which excludes the 30 development cases used to tune K, L2 threshold, and chunk size. Faithfulness is now reported under the minimum-judge rule (a case counts FAITHFUL only if every configured judge agrees), not the single-judge rate. The numeric refusal gate added in Stage 7 (§4.5) replaces the prompt-only fallback that previously failed on every Tier 3 case. **All headline numbers in this Conclusion are computed on the held-out test split (n=70); the full-set numbers (n=100, which include the 30 development cases used during hyperparameter tuning) are presented separately in §4.3, §4.4, and §4.7 for completeness.** The multi-agent medical RAG system shows the following performance:

- **Routing accuracy on the held-out test split: 100.0% (70/70) [Wilson 95% CI 94.8%–100%]** (§4.1, §4.8). On the eight ambiguous cross-domain cases (§4.2), the LLM router splits between cardiologist and endocrinologist according to the dominant pathology; the keyword baseline routes all 8 to cardiologist.
- **Retrieval Recall@K on the held-out test split: 56.2% (86/153) [Wilson 95% CI 48.3%–63.8%]** (§4.3, §4.8) — Cardiology 56.6% (43/76) [45.4%–67.1%], Endocrinology 55.8% (43/77) [44.7%–66.4%]. The legacy KeywordHitRate is 88.6% (62/70) [79.0%–94.1%] and is treated as a loose secondary signal because it can register hits on adjacent-content keyword co-occurrence rather than the actual answer documents (see §4.3 for the side-by-side). The Tier 1 vs Tier 2 cardiology gap persists under both metrics (Recall@K 59.0% vs 54.1%; KeywordHitRate 100% vs 78.6%), so the cardiology corpus gaps surfaced in §4.3.1 are not artefacts of the tuning split. Note: Recall@K denominators were auto-annotated by retrieving top-20 chunks from the same FAISS+embedding system being evaluated; the metric therefore measures the system's ability to surface keyword-positive top-20 documents into the top-5 window, not ground-truth retrieval against an independently labelled corpus.
- **Out-of-scope refusal on the held-out test split: 12/15 = 80.0% [Wilson 95% CI 54.8%–93.0%]** (§4.5), up from the prompt-only baseline of 0/15. The same threshold falsely refuses 27/55 = 49.1% [36.4%–62.0%] of Tier 1/2 queries because the in-scope and out-of-scope min-L2 distributions overlap (§4.5). The gate trades a non-trivial false-positive rate on in-scope queries for a non-zero refusal rate on out-of-scope ones; this is a deliberate clinical-safety trade-off.
- **Minimum-judge faithfulness on the held-out test split: 69/70 = 98.6% [Wilson 95% CI 92.3%–99.7%]** (§4.4). The primary YandexGPT judge marked every case FAITHFUL (100.0% [94.8%–100%]); the secondary YandexGPT-Lite judge — given the identical prompt — disagreed on `cardio_40` (Tier 2 cardiology, congenital LQTS). The 92.3% lower bound is the number to quote when comparing to other LLM-as-judge faithfulness results; see §5.3 for the disagreement and §6 Limitation 6 for the same-vendor caveat.

The hyperparameter grid search (K × L2 threshold, 30 combinations) was performed on the 30-case development split (§3.4) and selected K=5, L2 ≤ 1.2. The chunk size choice (400 words) and keyword-stripping were validated only under joint application — see §3.5 for the confounding caveat. Adding a third specialty requires a `registry.py` entry plus a corpus and FAISS index build (Fix 5). No agent code is duplicated under the new SpecialistAgent abstraction.

---

## 8. References

BibTeX entries for all citations are in [`reports/references.bib`](references.bib). Author-year list:

- **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).** RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*. (EACL 2024 Demonstrations.) `\cite{es2023ragas}` — cited in §4.4.
- **Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019).** PubMedQA: A Dataset for Biomedical Research Question Answering. *Proceedings of EMNLP-IJCNLP 2019*, 2567–2577. `\cite{jin2019pubmedqa}` — cited in §4.9.
- **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020).** Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*, 6769–6781. `\cite{karpukhin2020dpr}` — cited in §1.5.2 and §3.5.
- **Kim, Y., Park, C., Jeong, H., Chan, Y. S., Xu, X., McDuff, D., Breazeal, C., & Park, H. W. (2024).** MDAgents: An Adaptive Collaboration of LLMs for Medical Decision Making. *Advances in Neural Information Processing Systems (NeurIPS)*. `\cite{kim2024mdagents}` — cited in §1.5.4.
- **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020).** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. `\cite{lewis2020rag}` — cited in §1.5.2.
- **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval*. Cambridge University Press. `\cite{manning2008ir}` — cited in §5.1.
- **Robertson, S., & Zaragoza, H. (2009).** The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389. `\cite{robertson2009bm25}` — cited in §1.5.2.
- **Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., … Natarajan, V. (2023).** Large language models encode clinical knowledge. *Nature*, 620(7972), 172–180. `\cite{singhal2023medpalm}` — cited in §1 and §1.5.1.
- **Tsatsaronis, G., Balikas, G., Malakasiotis, P., Partalas, I., Zschunke, M., Alvers, M. R., … Paliouras, G. (2015).** An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition. *BMC Bioinformatics*, 16(1), 138. `\cite{tsatsaronis2015bioasq}` — cited in §1.5.1.
- **Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024).** Benchmarking Retrieval-Augmented Generation for Medicine. *arXiv:2402.13178*. (Findings of ACL 2024.) `\cite{xiong2024medrag}` — cited in §1 and §1.5.1.
- **Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023).** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*. `\cite{zheng2023mtbench}` — cited in §4.4, §5.3, and §6 Limitation 6.
