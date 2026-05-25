# Multi-Agent Medical RAG System

A clinical-decision-support assistant powered by a multi-agent Retrieval-Augmented Generation pipeline. Each natural-language clinical query is routed to a specialist agent — cardiology, endocrinology, gastroenterology, or infectiology — which retrieves evidence from its own curated knowledge base and generates a structured, grounded response.

**Disclaimer.** Academic and informational use only. Does not provide clinical diagnoses, prescribe treatments, or replace the judgment of a licensed physician.

## Contributors

| Contributor | GitHub | Contribution |
|---|---|---|
| Suvorova A. | [@Lunciare](https://github.com/Lunciare) | Core architecture, RAG pipeline, evaluation infrastructure, report |
| Murzova S. | [@Angry-Jupiter](https://github.com/Angry-Jupiter) | Gastroenterology and infectiology knowledge bases, specialist registry entries |

## Overview

A single generalist LLM tends to hallucinate over heterogeneous medical literature; a single retrieval index across all specialties dilutes precision. This system addresses both by (1) classifying the query's medical domain and dispatching it to a domain-specific agent, and (2) constraining generation to chunks retrieved from that agent's own FAISS + BM25 indices.

Four specialists are registered and evaluated end-to-end:

| Specialty | Corpus chunks | Notes |
|---|---:|---|
| Cardiology | 7,730 | Guidelines, textbooks, case reports, handbooks, articles |
| Endocrinology | 37,791 | Same categories |
| Gastroenterology | 8,670 | 354 PDF-extraction-artifact chunks filtered during indexing |
| Infectiology | 7,476 | 236 PDF-extraction-artifact chunks filtered during indexing |

Adding a new specialty requires one entry in `agents/registry.py`, a corpus, and an index build — the evaluation pipeline iterates the registry, so no eval-script changes are needed.

## Architecture

```
User Query
    |
    v
Orchestrator
    |-- Safety gate (regex): intercepts emergencies, prescription requests
    |-- LLM routing: classifies query into one of 4 specialties
    |
    v
Specialist Agent (parameterised from agents/registry.py)
    |-- FAISS similarity search (K=5, L2 <= 1.2)
    |-- Out-of-scope refusal gate (numeric L2 threshold)
    |-- Keyword-stripped chunk retrieval
    |-- LLM generation (temperature=0, grounded-only prompt)
    |
    v
Structured Clinical Response
    |-- Clinical Summary
    |-- Evidence-Based Insights
    |-- Limitations
```

All tuneable parameters (K, L2 threshold, chunk size, model names, gate thresholds) live in `multi-agent_system/settings.py`.

## Key Design Choices

- **Embedding model.** Yandex `text-search-doc/latest` (asymmetric bi-encoder, 256 dim); queries use `text-search-query/latest`. Vectors are pre-normalised by the API, so L2 and cosine produce identical rankings.
- **Keyword stripping.** Source chunks carry a `KEYWORDS:` header used for the TF-IDF baseline. The header is removed before embedding to avoid semantic distortion; it is preserved in metadata for hybrid retrieval.
- **Retrieval thresholds.** K=5 and L2 ≤ 1.2, tuned via grid search on the dev split. Larger K increases hit rate but degrades faithfulness by flooding the prompt with noisy context.
- **Generation.** Specialists run at `temperature=0.0` with a system prompt that prohibits facts not in the retrieved context. The "Insufficient evidence" fallback rule sits at the end of the prompt to exploit recency bias.
- **Out-of-scope refusal.** A numeric gate on minimum L2 distance refuses queries whose nearest chunk falls outside the in-corpus distribution.
- **Multi-judge faithfulness evaluation.** Three judges (`yandexgpt/latest`, `yandexgpt-lite/latest`, and cross-vendor `openai/gpt-oss-120b:free` via OpenRouter) score every response; a case is counted faithful only if all three agree, breaking the same-family circularity flagged by Zheng et al. 2023.

## Evaluation

Validated on a 200-case golden dataset (50 cases per specialty), partitioned into a 60-case dev split (used for hyperparameter tuning) and a 140-case held-out test split. Each specialty's 50 cases span three tiers: Tier 1 *Core* (in-corpus, prototypical), Tier 2 *Peripheral* (in-corpus, edge cases), Tier 3 *Out-of-scope* (deliberately outside the corpus, testing the refusal gate).

Held-out test results (n=140; 35 cases per specialty):

| Metric                       | Cardiology | Endocrinology | Gastroenterology | Infectiology | Overall |
|------------------------------|-----------:|--------------:|-----------------:|-------------:|--------:|
| Routing accuracy             | 100.0%     | 97.1%         | 88.6%            | 97.1%        | **95.7%** |
| Recall@5 (Tier 1)            | 59.0%      | 60.6%         | 52.8%            | 50.0%        | **55.7%** |
| Recall@5 (Tier 2)            | 54.1%      | 52.3%         | 60.0%            | 62.5%        | **56.8%** |
| Faithfulness (3-judge min)   | 94.3%      | 94.3%         | 100.0%           | 96.7%        | **96.2%** |
| Tier 3 refusal rate          | 0.0%       | 0.0%          | 42.9%            | 42.9%        | **20.7%** |
| Tier 1/2 false-positive rate | 11.1%      | 0.0%          | 10.7%            | 46.4%        | **17.1%** |

Recall@5 is reported for Tiers 1 and 2 only; Tier 3 cases have empty gold-source sets by design and are scored on refusal rate instead. Full Wilson 95% CIs, per-judge breakdowns, methodology, and failure analysis live in `reports/report_final.md`.

## Setup

Requires Python 3.11+ and a [Yandex Cloud](https://console.yandex.cloud/) account with Foundation Models API access (`text-search-doc/latest`, `yandexgpt/latest`).

```bash
git clone https://github.com/Lunciare/Multi-Agent-NN-Medicine.git
cd Multi-Agent-NN-Medicine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp multi-agent_system/.env.example multi-agent_system/.env
# edit .env: YANDEX_API_KEY, YANDEX_PROJECT_ID
playwright install chromium
```

## Building the Indices

The 60k+ chunked `.txt` sources under `data/processed/` are tracked in git; the binary FAISS / BM25 indices are not. After cloning, build them locally:

```bash
cd multi-agent_system
python build_index.py --specialty cardiologist
python build_index.py --specialty endocrinologist
python build_index.py --specialty gastroenterologist
python build_index.py --specialty infectionist
python build_bm25_index.py --specialty all
```

FAISS builds save progress every 500 chunks and resume after interruption. `build_index.py` filters chunks whose mean word length exceeds 15 characters (PDF-extraction artifacts — concatenated text without inter-word spaces) before embedding.

## Usage

Launch the Gradio interface (typically at `http://127.0.0.1:7860`):

```bash
cd multi-agent_system
python main.py
```

## Evaluation Suite

Pytest (fast, mocked, no API key required):

```bash
python -m pytest tests/ -v
```

Full evaluation pipeline (`--split test` for held-out results, `--split dev` for tuning):

```bash
cd multi-agent_system
python tests/evaluate_routing.py --split test
python tests/evaluate_retrieval.py --split test
python tests/evaluate_chunk_relevance.py --split test
python tests/evaluate_generation.py --split test --mode multi_judge
python tests/inspect_judge_disagreements.py ../reports/faithfulness_multijudge_raw_YYYY-MM-DD.csv
python tests/tune_retrieval.py
python tests/tune_chunk_size.py
```

The multi-judge faithfulness run requires `SECONDARY_JUDGE_PROVIDER` (and optionally `TERTIARY_JUDGE_PROVIDER`) in `.env`; see the `.env.example` template.

## Repository Structure

```
multi-agent-medical-rag/
├── data/
│   ├── raw/
│   └── processed/
├── multi-agent_system/
│   ├── agents/
│   ├── tests/
│   ├── build_index.py
│   ├── build_bm25_index.py
│   ├── orchestrator.py
│   ├── embeddings.py
│   ├── judges.py
│   ├── refusal_gate.py
│   ├── settings.py
│   └── main.py
├── tests/
├── scripts/
│   └── data_processing/
├── reports/
│   ├── report_final.md
│   └── archive/
└── requirements.txt
```
