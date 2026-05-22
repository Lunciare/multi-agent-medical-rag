# Multi-Agent Medical RAG System

A Clinical Decision Support Assistant powered by a multi-agent Retrieval-Augmented Generation (RAG) architecture. A clinical query is routed to a specialist agent, which retrieves evidence from a curated knowledge base and generates a structured, grounded clinical narrative.

**Disclaimer:** Academic and informational use only. Does not provide clinical diagnoses, prescribe treatments, or replace the judgment of a licensed physician.

## Contributors

| Contributor | GitHub | Contribution |
|---|---|---|
| Suvorova A. | [@Lunciare](https://github.com/Lunciare) | Core system architecture, RAG pipeline, evaluation infrastructure, report |
| Murzova S. | [@Angry-Jupiter](https://github.com/Angry-Jupiter) | Gastroenterology and infectiology knowledge base, specialist registry entries |

## Overview

The system accepts a natural-language clinical query, classifies its medical domain, and delegates it to a specialist agent. Each agent retrieves relevant chunks from its FAISS vector index and generates a response constrained to the retrieved evidence.

Four specialists are registered and evaluated end-to-end on a 200-case golden dataset (50 cases per specialty):

- **Cardiologist** — 7,730 chunks (Guidelines, Textbooks, Case Reports, Handbooks, Articles)
- **Endocrinologist** — 37,791 chunks across the same categories
- **Gastroenterologist** — 8,670 chunks (Articles, Cases, Guidelines, Handbooks, Textbooks); 354 PDF-extraction-artifact chunks skipped during indexing — see `reports/corpus_stats_2026-05-21.md`
- **Infectionist** — 7,476 chunks across the same five categories; 236 PDF-extraction-artifact chunks skipped during indexing

For prior work this project builds on and differs from, see [report §1.5 — Related Work](reports/report_final.md#15-related-work).

## Architecture

```
User Query
    |
    v
Orchestrator (orchestrator.py)
    |-- Safety gate (regex): intercepts emergency phrases, prescription requests
    |-- LLM routing call: classifies query domain (cardiology / endocrinology / gastroenterology / infectiology)
    |
    v
Specialist Agent (agents/specialist.py, parameterised by agents/registry.py)
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

Validated against a golden dataset of 200 clinical cases across three tiers and four specialties (50 cases per specialty):

| Tier | Label        | Cardiology | Endocrinology | Gastroenterology | Infectiology | Total |
|------|--------------|-----------:|--------------:|-----------------:|-------------:|------:|
| 1    | Core         | 27         | 27            | 27               | 27           | 108   |
| 2    | Peripheral   | 14         | 16            | 15               | 15           | 60    |
| 3    | Out-of-scope | 9          | 7             | 8                | 8            | 32    |

Tier 3 tests the "Insufficient evidence" safety fallback, not retrieval quality. Recall@5 is reported for Tier 1 + Tier 2 only (Tier 3 cases have `gold_sources=[]` by design); for Tier 3, the headline metric is the refusal rate of the numeric out-of-scope gate. **All numbers below are from the held-out test split (n=140 = 35 cases per specialty, cases 16–50).**

| Metric                       | Cardiology                  | Endocrinology              | Gastroenterology            | Infectiology               | Overall                              |
|------------------------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------------------|
| Routing Accuracy (all tiers) | 100.0% (35/35) [90.1%–100.0%] | 97.1% (34/35) [85.5%–99.5%] | 88.6% (31/35) [74.0%–95.5%] | 97.1% (34/35) [85.5%–99.5%] | **95.7% (134/140) [91.0%–98.0%]**     |
| Recall@5 (T1, pooled gold)   | 59.0% (23/39) [43.4%–72.9%]   | 60.6% (20/33) [43.7%–75.3%] | 52.8% (19/36) [37.0%–68.0%] | 50.0% (16/32) [33.6%–66.4%] | **55.7% (78/140) [47.4%–63.7%]**      |
| Recall@5 (T2, pooled gold)   | 54.1% (20/37) [38.4%–69.0%]   | 52.3% (23/44) [37.9%–66.2%] | 60.0% (21/35) [43.6%–74.4%] | 62.5% (20/32) [45.3%–77.1%] | **56.8% (84/148) [48.8%–64.4%]**      |
| Faithfulness (min-judge)     | 97.1% (34/35) [85.5%–99.5%]   | 100.0% (35/35) [90.1%–100.0%] | 100.0% (32/32) [89.3%–100.0%] | 100.0% (30/30) [88.7%–100.0%] | **99.2% (131/132) [95.8%–99.9%]**     |
| Tier 3 Refusal Rate (gate)   | 0.0% (0/8) [0.0%–32.4%]      | 0.0% (0/7) [0.0%–35.4%]    | 42.9% (3/7) [15.8%–75.0%]   | 42.9% (3/7) [15.8%–75.0%]   | **20.7% (6/29) [9.8%–38.4%]**         |
| Tier 1/2 FP Rate (gate)      | 11.1% (3/27) [3.9%–28.1%]    | 0.0% (0/28) [0.0%–12.1%]   | 10.7% (3/28) [3.7%–27.2%]   | 46.4% (13/28) [29.5%–64.2%] | **17.1% (19/111) [11.2%–25.2%]**      |

> **Footnote.** All numbers from the held-out test split (n=140; see [report §4.8](reports/report_final.md#48-held-out-test-set-results-n140)). For dev-set results used during hyperparameter tuning see [report §4.3–§4.7](reports/report_final.md#43-retrieval-hit-rate). Recall@5 denominators are pooled gold-doc Bernoulli (each Tier 1/2 case contributes 1–3 gold-doc trials). Faithfulness (min-judge) counts a case FAITHFUL only if both Yandex judges (`yandexgpt/latest` and `yandexgpt-lite/latest`) agree; the single disagreement is `cardio_40` (Tier 2 cardiology) — unchanged from the prior 2-specialty baseline. Refusal Rate uses the Stage 39 re-tuned numeric out-of-scope gate `L2_REJECT_MIN = 1.020` ([report §4.5](reports/report_final.md#45-out-of-scope-refusal-gate)); both targets (≥80% T3 recall, ≤5% T1/T2 FP) are unmet at this threshold because the in-corpus / out-of-corpus min-L2 distributions overlap heavily at the 4-specialty scale — a two-stage gate is the proposed fix ([§6 Limitation 8](reports/report_final.md#6-limitations)).

### Limitations

- **Small sample size (L1).** n=140 test cases gives Wilson 95% CIs of ±5–25 pp depending on the tier; the per-tier point estimates carry substantial uncertainty. See [report §6 Limitation 1](reports/report_final.md#6-limitations).
- **Corpus coverage gaps surfaced by name (L2).** Six Tier 1/2 cases have `gold_sources = []` because the corpus's top-20 retrieval contains no keyword-matching documents: pre-existing `cardio_35` (temporary pacing for STEMI block) and `endo_46` (hypoglycaemia unawareness); plus new `gastro_39` (haemochromatosis), `gastro_44` (Zollinger-Ellison), `infect_21` (HSV encephalitis), `infect_39` (prosthetic joint infection). `gastro_37` (viral gastroenteritis) and `infect_14` (C. difficile colitis) were retired from the list on 2026-05-22 — keyword-set mismatches, not corpus gaps — and now have 3 gold docs each. See [report §4.3.1](reports/report_final.md#431-tier-2-corpus-coverage-audit) and [§6 Limitation 2](reports/report_final.md#6-limitations).
- **Bounded to 4 specialties (L3).** Cardiology, endocrinology, gastroenterology, and infectiology are the only registered domains. Adding another specialty is a single `agents/registry.py` entry plus a corpus + FAISS/BM25 index build; the evaluation pipeline iterates `AGENT_REGISTRY.keys()` (generalised at Stage 39), so no eval-script changes are required. See [report §6 Limitation 3](reports/report_final.md#6-limitations).
- **Single-language corpus (L5).** All source documents are in English. The §4.2.1 adversarial set scores 100% on Russian / French / Spanish *surface vocabulary* (routing only), but retrieval and generation have not been validated on non-English queries. See [report §6 Limitation 5](reports/report_final.md#6-limitations).
- **Same-vendor judge bias on faithfulness (L6).** Both judges are Yandex models; the 99.2% min-judge rate is an upper bound under the methodology characterised by [Zheng et al. 2023](reports/report_final.md#8-references). A cross-vendor judge slot (`TERTIARY_JUDGE_PROVIDER`) is implemented but not configured here. See [report §5.3](reports/report_final.md#53-epistemic-bounds-of-same-family-evaluation) and [§6 Limitation 6](reports/report_final.md#6-limitations).
- **Auto-annotation circularity in Recall@K (L7).** Gold-source annotations were produced by `tests/annotate_gold_sources.py --auto`, which selects up to three documents per case from the *same* FAISS+embedding system's top-20 retrieval. Recall@K therefore measures "fraction of keyword-positive top-20 docs that surface at K=5", not "fraction of ground-truth docs retrieved". A human- or external-benchmark-curated annotation pass would break the circularity. See [report §6 Limitation 7](reports/report_final.md#6-limitations).
- **Out-of-scope refusal — both targets unmet at 4-specialty scale (L8).** The Stage 7 single-threshold numeric gate cannot simultaneously hit ≥80% T3 recall AND ≤5% T1/T2 FP on the 4-specialty corpus — the new specialties' in-corpus L2 distribution overlaps the out-of-corpus distribution heavily. The Stage 39 re-tune chose the best-F1 fallback (`L2_REJECT_MIN = 1.020`, 20.7% T3 recall, 17.1% T1/T2 FP). A two-stage gate (L2 pre-filter + LLM-as-classifier confirmer) is now urgent. See [report §4.5](reports/report_final.md#45-out-of-scope-refusal-gate) and [§6 Limitation 8](reports/report_final.md#6-limitations).

Evaluation scripts: `multi-agent_system/tests/`.

## Repository Structure

```
Multi-Agent-NN-Medicine/
├── data/processed/
│   ├── cardiology/                # 7,730 chunks + FAISS + BM25 indices
│   ├── endocrinology/             # 37,791 chunks + FAISS + BM25 indices
│   ├── gastroenterologist/        # 8,670 chunks + FAISS + BM25 indices
│   └── infection/                 # 7,476 chunks + FAISS + BM25 indices
├── multi-agent_system/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── specialist.py
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
│   ├── build_index.py             # Build a FAISS index from AGENT_REGISTRY
│   ├── build_bm25_index.py        # Build the BM25 baseline index
│   ├── orchestrator.py
│   ├── embeddings.py
│   ├── judges.py                  # Multi-judge faithfulness evaluator
│   ├── refusal_gate.py            # Out-of-scope refusal gate
│   ├── settings.py                # Hyperparameters and API config
│   └── main.py                    # Gradio interface
├── tests/                         # Pytest suite (safety, error handling, integration, playwright, retrieval regression)
├── scripts/data_processing/       # One-time corpus preparation scripts
├── reports/
│   ├── report_final.md                       # Canonical end-to-end evaluation
│   ├── report_independent_audit.md           # External audit (drives Stage-40 work)
│   ├── report_stage_indices_built.md         # Stage-38: gastro + infect FAISS+BM25 built
│   ├── report_stage_dataset_extended.md      # Stage-38: 200-case golden dataset
│   ├── report_stage_full_integration.md      # Stage-39: 4-spec eval pipeline
│   ├── report_stage_new_agents.md            # Stage-39: registry + routing prompt
│   ├── corpus_stats_2026-05-21.md            # Per-corpus chunk inventory
│   ├── failure_analysis.md                   # Per-case failure breakdown
│   ├── hyperparameter_grid.csv               # K × L2 grid search
│   ├── refusal_gate_grid.csv                 # Stage-39 refusal-gate threshold sweep
│   ├── faithfulness_multijudge_2026-05-21.{md,csv}  # Multi-judge test-split run
│   ├── routing_evaluation_2026-05-21_*.md    # Latest LLM + adversarial routing runs
│   └── archive/                              # Pre-Stage-39 per-stage reports & logs
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

### Building the FAISS and BM25 indices

The 64,823 chunked source `.txt` files under `data/processed/` are tracked in git, but the binary FAISS / BM25 index files are **not** — `.gitignore` excludes `data/processed/**/faiss_index/` and `*.pkl`. After cloning, you must build the indices locally before the system can answer queries. The full four-specialty build:

```bash
cd multi-agent_system

# 1. FAISS vector indices (one per specialty; hits the Yandex embedding API)
python build_index.py --specialty cardiologist        # ~40 min for 7,730 chunks
python build_index.py --specialty endocrinologist     # ~2.5 hours for 37,791 chunks
python build_index.py --specialty gastroenterologist  # ~50 min for 8,670 chunks
python build_index.py --specialty infectionist        # ~45 min for 7,476 chunks

# 2. BM25 keyword indices (required for routing + hybrid retrieval; no API calls, <1 min total)
python build_bm25_index.py --specialty all
```

All four FAISS builds save progress every 500 chunks and resume after interruptions. `build_index.py` also filters chunks whose mean word length exceeds 15 characters (PDF-extraction artifacts — concatenated text without inter-word spaces) before embedding; this drops ~5 % of the gastro / infect corpora and 0 % of cardio / endo.

> **Follow-up for maintainers.** If clone-to-run matters more than repo size, consider committing the FAISS / BM25 index binaries via [git LFS](https://git-lfs.com/) so first-time users skip the ~5 h rebuild and avoid the embedding-API spend.

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
