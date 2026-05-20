# Multi-Agent Medical RAG System

A Clinical Decision Support Assistant powered by a multi-agent Retrieval-Augmented Generation (RAG) architecture. A clinical query is routed to a specialist agent, which retrieves evidence from a curated knowledge base and generates a structured, grounded clinical narrative.

**Disclaimer:** Academic and informational use only. Does not provide clinical diagnoses, prescribe treatments, or replace the judgment of a licensed physician.

## Overview

The system accepts a natural-language clinical query, classifies its medical domain, and delegates it to a specialist agent. Each agent retrieves relevant chunks from its FAISS vector index and generates a response constrained to the retrieved evidence.

Two specialists are implemented:

- **Cardiologist** — 7,730 chunks (Guidelines, Textbooks, Case Reports, Handbooks, Articles)
- **Endocrinologist** — 37,791 chunks across the same categories

## Architecture

```
User Query
    |
    v
Orchestrator (orchestrator.py)
    |-- Safety gate (regex): intercepts emergency phrases, prescription requests
    |-- LLM routing call: classifies query domain (cardiology / endocrinology)
    |
    v
Specialist Agent (agents/cardiologist.py | agents/endocrinologist.py)
    |-- FAISS similarity search (K=5, L2 <= 1.2)
    |-- Keyword-stripped chunk retrieval
    |-- Yandex LLM generation (temperature=0.0)
    |
    v
Structured Clinical Response
    |-- Clinical Summary
    |-- Evidence-Based Insights (grounded in retrieved context)
    |-- Limitations
```

All tuneable parameters (K, L2 threshold, chunk size, model names) live in `multi-agent_system/settings.py`.

## Evaluation Results

Validated against a golden dataset of 100 clinical cases across three tiers:

| Tier | Label        | Cardiology | Endocrinology | Total |
|------|--------------|------------|---------------|-------|
| 1    | Core         | 27         | 27            | 54    |
| 2    | Peripheral   | 14         | 16            | 30    |
| 3    | Out-of-scope | 9          | 7             | 16    |

Tier 3 tests the "Insufficient evidence" safety fallback, not retrieval quality. Hit Rate and Faithfulness are reported for Tier 1 and Tier 2 only.

> **TODO — fill in current results from `reports/report_final.md`:**

| Metric             | T1 Cardiology | T1 Endocrinology | T2 Cardiology | T2 Endocrinology | Overall |
|--------------------|---------------|------------------|---------------|------------------|---------|
| Routing Accuracy   | TBD           | TBD              | TBD           | TBD              | TBD     |
| Hit Rate           | TBD           | TBD              | TBD           | TBD              | TBD     |
| Faithfulness       | TBD           | TBD              | TBD           | TBD              | TBD     |

Evaluation scripts: `multi-agent_system/tests/`.

## Repository Structure

```
Multi-Agent-NN-Medicine/
├── data/processed/
│   ├── cardiology/                # 7,730 chunks + FAISS index
│   └── endocrinology/             # 37,791 chunks + FAISS index
├── multi-agent_system/
│   ├── agents/
│   │   ├── base.py
│   │   ├── cardiologist.py
│   │   └── endocrinologist.py
│   ├── tests/
│   │   ├── evaluate_retrieval.py
│   │   ├── evaluate_chunk_relevance.py
│   │   ├── evaluate_generation.py
│   │   ├── evaluate_routing.py
│   │   ├── evaluate_routing_baseline.py
│   │   ├── tune_retrieval.py
│   │   ├── tune_chunk_size.py
│   │   ├── save_test_vectors.py     # One-time: writes test_vectors.npy
│   │   └── data/
│   │       ├── golden_dataset.json
│   │       ├── ambiguous_cases.json
│   │       ├── test_vectors.npy        # Pre-saved query embeddings
│   │       └── test_vector_labels.json
│   ├── build_cardio_faiss.py      # Build cardiology FAISS index from scratch
│   ├── build_endo_faiss.py        # Build endocrinology FAISS index from scratch
│   ├── orchestrator.py
│   ├── embeddings.py
│   ├── settings.py                # Hyperparameters and API config
│   └── main.py                    # Gradio interface
├── tests/                         # Pytest suite (safety, error handling, integration, playwright, retrieval regression)
├── scripts/data_processing/       # One-time corpus preparation scripts
├── reports/
│   ├── report_final.md
│   ├── report_stage_2.md
│   ├── report_stage_3.md
│   ├── failure_analysis.md
│   ├── hyperparameter_grid.csv
│   ├── routing_evaluation_2026-05-11_16-06-10.md
│   └── archive/                   # Earlier routing-evaluation snapshots
└── requirements.txt
```

## Setup

Requires Python 3.11+ and a [Yandex Cloud](https://console.yandex.cloud/) account with Foundation Models API access (`text-search-doc/latest`, `yandexgpt/latest`).

```bash
git clone https://github.com/Lunciare/Multi-Agent-NN-Medicine.git
cd Multi-Agent-NN-Medicine

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp multi-agent_system/.env.example multi-agent_system/.env
# edit .env: set YANDEX_API_KEY and YANDEX_PROJECT_ID

# The pytest suite includes a Playwright browser test — install Chromium:
playwright install chromium
```

### Building the FAISS indices

Pre-built indices are committed. To rebuild from scratch (e.g., after changing chunk size or adding documents):

```bash
cd multi-agent_system
python build_cardio_faiss.py   # ~40 min for 7,730 chunks
python build_endo_faiss.py     # ~2.5 hours for 37,791 chunks
```

Both scripts save progress every 500 chunks and resume after interruptions.

### Running the system

```bash
cd multi-agent_system
python main.py
```

This launches a local Gradio server (typically at `http://127.0.0.1:7860`). On the first run without a cached index, it builds the FAISS index automatically; subsequent runs load it instantly.

## Running the Pytest Suite

The top-level `tests/` directory holds the CI test suite — fast, mocked, no API key required (with one setup step for retrieval regression):

```bash
python -m pytest tests/ -v
```

Coverage:
- `test_safety.py` — emergency / treatment-request gating.
- `test_error_handling.py` — input validation, OpenAI exception mapping, missing-corpus errors.
- `test_integration.py` — orchestrator construction, routing, end-to-end answer flow (all LLM calls mocked).
- `test_playwright.py` — headless Chromium smoke test (requires `playwright install chromium`).
- `test_retrieval_regression.py` — offline FAISS regression. For each of 10 canonical queries (5 cardiology + 5 endocrinology) it reads the pre-saved query embedding from `multi-agent_system/tests/data/test_vectors.npy`, runs it through the correct FAISS index, and diffs the live top-K=5 against `test_retrieval_snapshot.json`. Two assertions per query: (a) the *set* of retrieved `source_file`s must equal the snapshot's set, and (b) per-rank L2 drift must be < 0.1. Both fail with verbose diffs (added / removed source files, per-rank distance deltas). The earlier "≥1 chunk within `MAX_L2_DISTANCE`" sanity check is kept as a second test. Skips cleanly if either the `.npy` or the snapshot is missing.

### Regression testing

The retrieval regression test pins three things at once: the query embedder, the chunkification, and the FAISS index parameters (K, L2 threshold, ordering). After **any** change that can shift retrieval — re-chunking, re-embedding, swapping the embedder, changing `MAX_L2_DISTANCE`, etc. — the snapshot must be regenerated:

```bash
cd multi-agent_system
python tests/save_test_vectors.py --update-snapshot
```

This re-embeds the 10 canonical queries, runs each through the correct FAISS index, and overwrites `tests/data/test_retrieval_snapshot.json`. Commit the new snapshot together with the change that caused it. Without `--update-snapshot` the script preserves the existing snapshot file (it still refreshes `.npy` and `.json` if missing) so accidentally running the script never silently masks a regression.

To (re)generate `test_vectors.npy` for the first time (one-time, requires `YANDEX_API_KEY`):

```bash
cd multi-agent_system
python tests/save_test_vectors.py --update-snapshot
```

The resulting `.npy` and `.json` are committed to the repo so subsequent test runs are offline.

## Running Evaluations

```bash
cd multi-agent_system

# Smoke test — dataset integrity (no API calls, <30s)
python tests/evaluate_retrieval.py --smoke-test

# Routing Accuracy on held-out test split (~70 LLM calls)
python tests/evaluate_routing.py --split test

# Retrieval Hit Rate on held-out test split (no LLM calls)
python tests/evaluate_retrieval.py --split test

# Chunk Relevancy on held-out test split (LLM judge, ~70 calls)
python tests/evaluate_chunk_relevance.py --split test

# Faithfulness — multi-judge mode is now the default (~140–210 LLM calls)
# Requires SECONDARY_JUDGE_PROVIDER in .env; TERTIARY_JUDGE_PROVIDER is optional.
# Writes reports/faithfulness_multijudge_raw_$(date).csv and ..._$(date).md.
python tests/evaluate_generation.py --split test --mode multi_judge

# Faithfulness — single-judge legacy mode for backward compatibility
python tests/evaluate_generation.py --split test --mode yandex_only

# Inspect judge disagreements after a multi-judge run
python tests/inspect_judge_disagreements.py \
    ../reports/faithfulness_multijudge_raw_YYYY-MM-DD.csv

# Hyperparameter tuning — K and L2 threshold (dev split only; no LLM calls)
python tests/tune_retrieval.py

# Hyperparameter tuning — chunk size grid (dev split only; embedding API only)
python tests/tune_chunk_size.py
```

### Configuring the multi-judge faithfulness evaluator

`evaluate_generation.py` defaults to a multi-judge run that requires at least one
second judge to break the same-model circularity discussed in `report_final.md`
§5.3. Configure judges via environment variables (one per line in `.env`):

```bash
# Same vendor, different family — guaranteed to work with the existing Yandex API key
SECONDARY_JUDGE_PROVIDER=yandex:gpt://${YANDEX_PROJECT_ID}/yandexgpt-lite/latest

# Optional third judge (cross-vendor; signs up to whichever free-tier you can access)
# TERTIARY_JUDGE_PROVIDER=openrouter:meta-llama/llama-3.1-8b-instruct:free
# OPENROUTER_API_KEY=sk-or-v1-...

# Or, generic OpenAI-compatible endpoint
# TERTIARY_JUDGE_PROVIDER=http:https://api.example.com/v1/chat/completions
# SECONDARY_JUDGE_API_KEY=...
```

## Key Design Decisions

**Embedding model.** Yandex `text-search-doc/latest` (asymmetric bi-encoder, 256 dim). Query embeddings use `text-search-query/latest`. Vectors are pre-normalised by the API (norm ≈ 1.0), so L2 and cosine produce identical rankings.

**Keyword stripping.** Source chunks contain a `KEYWORDS:` header line used for TF-IDF indexing. This line is removed before embedding to prevent semantic distortion. Keywords are preserved in metadata for future hybrid search.

**Retrieval threshold.** K=5 and L2 ≤ 1.2, tuned via grid search on the golden dataset. Higher K (e.g., K=10) increases hit rate but degrades faithfulness by flooding the LLM with noisy context.

**Generation constraints.** Agents run at `temperature=0.0` with a system prompt that prohibits facts not in retrieved context. The "Insufficient evidence" fallback rule is at the end of the prompt to exploit recency bias.
