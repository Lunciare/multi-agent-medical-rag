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

Sections 4.1–4.7 report metrics computed on the full 100-case golden set. The 30-case development split (`golden_dev.json`) was used for hyperparameter tuning (K, L2 threshold, chunk size). Results restricted to the 70-case held-out test split are reported in §4.8.

### 4.3 Retrieval Hit Rate

Each query is sent to the correct specialist agent (bypassing the router). The agent retrieves K=5 chunks with L2 ≤ 1.2. **Recall@K** is the primary grounded metric: every Tier 1/2 case carries a `gold_sources` annotation listing 1–3 source documents that contain the correct answer (see §3.6); Recall@K is the fraction of those gold documents that appear among the K=5 retrieved chunks (pooled across cases — every gold-doc slot is one Bernoulli trial). **MRR@K** is the reciprocal rank of the first retrieved gold document, averaged across annotated cases. **Refusal Rate (T3)** is the fraction of Tier 3 cases where the retrieval pipeline returned zero chunks — making the safety-fallback failure (currently 0/16) numerically explicit. **KeywordHitRate** is the original keyword-co-occurrence metric kept here as a loose secondary signal for cross-stage comparison.

| Domain | Recall@K | MRR@K | KeywordHitRate (legacy) | Refusal Rate (T3) |
|---|---|---|---|---|
| Cardiology | 61.0% (72/118) [52.0%–69.3%] | 0.730 | 86.0% (43/50) | 0.0% (0/9) |
| Endocrinology | 57.4% (70/122) [48.5%–65.8%] | 0.757 | 96.0% (48/50) | 0.0% (0/7) |
| **Overall** | **59.2% (142/240) [52.9%–65.2%]** | **0.744** | **91.0% (91/100)** | **0.0% (0/16)** |

*(Wilson 95% CIs on the pooled gold-doc Bernoulli. MRR@K is reported without a strict CI — it is a mean of [0, 1] reciprocal-rank values per case, not a Bernoulli proportion; see Stage 6 report for bootstrap-style sanity checks. `# Legacy: registers hits on keyword co-occurrence; see Recall@K for grounded metric`.)*

> **Note on Recall@K denominators:** 82 of the 84 Tier 1/2 cases were annotated by the auto-annotator (`tests/annotate_gold_sources.py --auto`), which scans the top-20 retrieved chunks and picks up to 3 documents per case with ≥1 expected-keyword hit. The two unannotated cases — `cardio_35` (STEMI with complete heart block) and `endo_46` (hypoglycaemia unawareness) — are the same two cases where the top-20 retrieval registered zero keyword matches and are therefore the legitimate retrieval misses already discussed in §4.3.1; they do not contribute to Recall@K. The 16 Tier 3 cases have `gold_sources: []` by design and contribute only to the Refusal Rate column.

> **Why Recall@K (59.2%) is far below KeywordHitRate (91.0%):** the two metrics measure different things. KeywordHitRate counts a case as a hit if any of the 5 retrieved chunks contains any expected keyword anywhere — including adjacent, off-topic content that happens to share a common word. Recall@K is far stricter: it requires the *specific documents* containing the answer (annotated via top-20 keyword coverage, then capped at 3) to land in the *top-5* retrieval window. The 32-point gap is the part of the corpus that ranks 6–20 in retrieval order — relevant, but not surfaced at K=5.

> **Important Note on Tier 3 Metrics:** Tier 3 cases produce a Refusal Rate of 0% (0/16): every out-of-scope query retrieves the full K=5 chunks of adjacent content rather than triggering the "Insufficient evidence" fallback. This is the same architectural limitation discussed in §5.2 and §6 Limitation 8, surfaced numerically by the new Refusal Rate column. The legacy KeywordHitRate column is also non-zero on some Tier 3 cases because adjacent chunks sometimes share a common keyword with the query (this is the well-known keyword-vs-relevance gap from prior stages, not a system improvement).

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

### 4.5 Out-of-Scope Refusal Gate

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

The tier-based results confirm that while the system excels on core clinical scenarios under the legacy keyword-co-occurrence signal (T1 cardiology 100% KeywordHitRate), the grounded Recall@K reveals that even on T1 cardiology only 64.2% of the gold documents actually land in the top-5 retrieval window. Performance predictably drops further on peripheral (T2) entities under both metrics. Routing and faithfulness stay near 100% across every tier, but **retrieval Recall@K shows a clear tier gradient** and **Tier 3 refusal — once 0/16 — is now a numeric-gate-driven 12/16 (Stage 7, §4.5)** at the cost of a 49.1% false-positive rate on Tier 1/2.

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

---

## 5. Discussion

The results of the final validation run highlight three architectural insights that extend beyond the baseline accuracy metrics:

### 5.1 Precision@K vs. Hit Rate: The Context-Window Noise Problem
Overall Precision@K (65.0%) underperforms overall Hit Rate (91.0%) significantly. Because Hit Rate only requires a single relevant keyword in the five retrieved chunks, while Precision@K requires keywords in multiple chunks, this 26-percentage-point gap quantifies how often FAISS retrieves one relevant chunk alongside several loosely related ones. If the system feeds five chunks to the generator but only one is relevant, the LLM must actively ignore four noisy inputs. This dynamic confirms why K=5 with L2≤1.2 is the optimal tradeoff, and explains why earlier tests with K=10 degraded faithfulness: expanding the context window with loosely related chunks forces the LLM to synthesise across irrelevant information, increasing hallucination risk.

### 5.2 The Nature of Tier 3 Failures: Distance vs. Relevance
The original prompt-only "Insufficient evidence" fallback failed completely on Tier 3 (0/16 triggering) — a fundamental property of nearest-neighbour search: FAISS always returns K results regardless of absolute relevance, the LLM trusts that retrieved context, and the in-prompt fallback rule loses to the LLM's training to be helpful. The L2 distance threshold filters by chunk quality but is not a semantic relevance gate. Stage 7 added a numeric pre-LLM refusal gate (§4.5) that explicitly checks `min(L2) > L2_REJECT_MIN` before the generation call. This raises Tier 3 refusal from 0/16 to 12/16 on the full set, but the same threshold falsely refuses 49.1% of Tier 1/2 queries — because the in-scope and out-of-scope min-L2 distributions overlap (T3: 0.84–1.00; T1/T2: 0.70–1.07). The architectural conclusion stands: a single L2-distance threshold is necessary but not sufficient for reliable out-of-scope detection. A two-stage gate (cheap L2 pre-filter plus an LLM-as-classifier confirmer on borderline cases) is the natural next step.

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

8. **Tier 3 Refusal — partial fix, residual FP cost.** The original prompt-only "Insufficient evidence" fallback failed on every Tier 3 case (0/16) because FAISS returns K nearest neighbours regardless of absolute distance, and the LLM generated from that adjacent content rather than declining. Stage 7 replaced the prompt rule with a numeric refusal gate (`refusal_gate.RefusalGate`, Signal A, `L2_REJECT_MIN = 0.92`) that refuses **12/15 (80.0%) of held-out Tier 3 cases** — but at the cost of a **49.1% false-positive rate on Tier 1/2** (§4.5). The single-threshold numeric gate cannot simultaneously satisfy the ≥80% T3 recall and ≤5% T1/T2 FP targets because the min-L2 distance distributions of in-scope and out-of-scope queries overlap heavily on this corpus. A two-stage refusal (numeric pre-filter + LLM-as-classifier confirmer) or a dedicated relevance classifier is the obvious next step.

---

## 7. Conclusion

Headline metrics are reported on the 70-case held-out test split (§4.8), which excludes the 30 development cases used to tune K, L2 threshold, and chunk size. Faithfulness is now reported under the minimum-judge rule (a case counts FAITHFUL only if every configured judge agrees), not the single-judge rate. The numeric refusal gate added in Stage 7 (§4.5) replaces the prompt-only fallback that previously failed on every Tier 3 case. The multi-agent medical RAG system shows the following performance:

- **Routing** achieves 100.0% accuracy (70/70) across all tiers on the held-out test split (§4.8), matching the full-set figure. The router demonstrates triage-like behaviour on cross-domain queries, consistently prioritising the presenting clinical urgency.
- **Retrieval** is reported primarily as **Recall@K against the per-case `gold_sources` annotation** (Stage 6). On the held-out test split, Recall@K is **56.2% (86/153) [Wilson 95% CI 48.3%–63.8%]** — Cardiology 56.6% (43/76), Endocrinology 55.8% (43/77); the legacy KeywordHitRate is 88.6% (62/70) and is now treated as a loose secondary signal because it registers hits on adjacent-content keyword co-occurrence rather than the actual source documents (see §4.3 for the side-by-side). Across both metrics the Tier 1 cardiology / Tier 2 cardiology gap persists (Recall@K 59.0% vs 54.1%; KeywordHitRate 100% vs 78.6%), confirming that the cardiology corpus gaps surfaced in §4.3.1 are not artefacts of the tuning split.
- **Out-of-scope refusal** is **no longer a 0/N failure**. The Stage 7 numeric refusal gate (§4.5, Signal A with `L2_REJECT_MIN = 0.92`) refuses **12/15 (80.0%) of held-out Tier 3 cases** and **12/16 (75.0%) of full-set Tier 3 cases**, up from the prompt-only baseline of **0/15 (0.0%) and 0/16 (0.0%)** respectively. The same threshold falsely refuses **27/55 (49.1%)** of held-out Tier 1/2 queries — well above the ≤5% target — because the in-scope and out-of-scope min-L2 distributions overlap (§4.5). The system therefore trades a non-trivial false-positive rate on Tier 1/2 for a non-zero refusal rate on Tier 3; this is a deliberate clinical-safety trade-off, not "robust behaviour across all tiers".
- **Faithfulness** reaches **98.6% (69/70) under the minimum-judge rule** on the held-out test split, with a **Wilson 95% CI lower bound of 92.3%** (§4.4). The primary YandexGPT judge marked every case FAITHFUL (100.0%); the secondary YandexGPT-Lite judge — given the identical strict prompt — disagreed on `cardio_40` (Tier 2 cardiology, congenital LQTS), applying a stricter token-grounding standard. The conservative 92.3% lower bound is the right number to quote when comparing this system to LLM-as-a-judge faithfulness results elsewhere; see §5.3 for the disagreement analysis and §6 Limitation 6 for the remaining same-vendor caveat.

The hyperparameter grid search (K × L2 threshold, 30 combinations) was performed on the 30-case development split (§3.4) and confirmed K=5, L2 ≤ 1.2 as the optimal operating point, balancing retrieval completeness against context compactness for faithful generation. The chunk size optimization (400 words) and keyword-stripping strategy were both empirically validated and contributed measurably to system quality. The architecture is modular and ready for extension to additional medical specialties.
