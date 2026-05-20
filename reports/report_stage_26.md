# Stage 26 — README Refresh: Match Current Code

(Filename note: numbered as Stage 26 because Stages 23–25 are already
taken — Stage 23 = orchestrator dict-access migration, Stage 24 = wiring
`domain_scope` into routing prompt, Stage 25 = adversarial routing test
set. The task spec didn't name a target filename for this report; using
the next sequential number.)

## 1. What Was Changed

The README had drifted out of sync with the Stage 8 SpecialistAgent
refactor (deletion of `agents/cardiologist.py` + `agents/endocrinologist.py`
and the per-specialty `build_cardio_faiss.py` + `build_endo_faiss.py`
scripts, replaced by the registry-driven `agents/specialist.py` +
`agents/registry.py` and the unified `build_index.py`), and was also
missing two top-level modules that have landed since: `judges.py`
(multi-judge faithfulness — Stage 5) and `refusal_gate.py` (out-of-scope
numeric gate — Stage 7). This stage brings the README's three
architecture-describing surfaces (Architecture ASCII diagram, Repository
Structure tree, Building the FAISS indices commands) into one-to-one
correspondence with the current disk layout, with zero stale references
to deleted files.

## 2. Full Diff (`README.md`)

```diff
diff --git a/README.md b/README.md
index aa7ce68cb..d309bc9de 100644
--- a/README.md
+++ b/README.md
@@ -26,7 +26,7 @@ Orchestrator (orchestrator.py)
     |-- LLM routing call: classifies query domain (cardiology / endocrinology)
     |
     v
-Specialist Agent (agents/cardiologist.py | agents/endocrinologist.py)
+Specialist Agent (agents/specialist.py, parameterised by agents/registry.py)
     |-- FAISS similarity search (K=5, L2 <= 1.2)
     |-- Keyword-stripped chunk retrieval
     |-- Yandex LLM generation (temperature=0.0)
@@ -80,9 +80,10 @@ Multi-Agent-NN-Medicine/
 │   └── endocrinology/             # 37,791 chunks + FAISS index
 ├── multi-agent_system/
 │   ├── agents/
+│   │   ├── __init__.py
 │   │   ├── base.py
-│   │   ├── cardiologist.py
-│   │   └── endocrinologist.py
+│   │   ├── registry.py
+│   │   └── specialist.py
 │   ├── tests/
 │   │   ├── evaluate_retrieval.py
 │   │   ├── evaluate_chunk_relevance.py
@@ -97,10 +98,12 @@ Multi-Agent-NN-Medicine/
 │   │       ├── ambiguous_cases.json
 │   │       ├── test_vectors.npy        # Pre-saved query embeddings
 │   │       └── test_vector_labels.json
-│   ├── build_cardio_faiss.py      # Build cardiology FAISS index from scratch
-│   ├── build_endo_faiss.py        # Build endocrinology FAISS index from scratch
+│   ├── build_index.py             # Build a FAISS index from AGENT_REGISTRY
+│   ├── build_bm25_index.py        # Build the BM25 baseline index
 │   ├── orchestrator.py
 │   ├── embeddings.py
+│   ├── judges.py                  # Multi-judge faithfulness evaluator
+│   ├── refusal_gate.py            # Out-of-scope refusal gate
 │   ├── settings.py                # Hyperparameters and API config
 │   └── main.py                    # Gradio interface
 ├── tests/                         # Pytest suite (safety, error handling, integration, playwright, retrieval regression)
@@ -141,8 +144,8 @@ Pre-built indices are committed. To rebuild from scratch (e.g., after changing c

 ```bash
 cd multi-agent_system
-python build_cardio_faiss.py   # ~40 min for 7,730 chunks
-python build_endo_faiss.py     # ~2.5 hours for 37,791 chunks
+python build_index.py --specialty cardiologist     # ~40 min for 7,730 chunks
+python build_index.py --specialty endocrinologist  # ~2.5 hours for 37,791 chunks
 ```

 Both scripts save progress every 500 chunks and resume after interruptions.
```

Three edit blocks across the file:

- **Block 1 (Architecture ASCII diagram, line 29).** Replaced the
  `agents/cardiologist.py | agents/endocrinologist.py` line with
  `agents/specialist.py, parameterised by agents/registry.py`.
- **Block 2 (Repository Structure tree, lines 82–110).** Updated the
  `agents/` subtree to list `__init__.py`, `base.py`, `registry.py`,
  `specialist.py` (matches the actual `ls multi-agent_system/agents/`);
  replaced the two `build_cardio_faiss.py` / `build_endo_faiss.py` lines
  with `build_index.py` + `build_bm25_index.py`; added `judges.py` and
  `refusal_gate.py` rows between `embeddings.py` and `settings.py`.
- **Block 3 (Building the FAISS indices, lines 144–146).** Replaced the
  two `python build_cardio_faiss.py` / `python build_endo_faiss.py`
  invocations with `python build_index.py --specialty cardiologist` /
  `--specialty endocrinologist`, keeping the original timing annotations
  unchanged.

The narrative claim that follows the build block — "Both scripts save
progress every 500 chunks and resume after interruptions" — still holds
under the unified `build_index.py` (see `SAVE_EVERY_N = 500` in
`multi-agent_system/build_index.py:38` and the resume-from-cache path),
so it was left intact.

## 3. Grep Smoke-Test Outputs (zero stale references)

Commands (verbatim from the spec):

```
! grep -E "build_cardio_faiss|build_endo_faiss" README.md
! grep -E "agents/cardiologist\.py|agents/endocrinologist\.py" README.md
grep -c "build_index.py" README.md   # expected: >=2
grep -c "agents/specialist.py" README.md   # expected: >=1
```

Outputs:

```
--- grep for build_cardio_faiss|build_endo_faiss (must be empty) ---
  EXIT: 1
--- grep for agents/cardiologist.py|agents/endocrinologist.py (must be empty) ---
  EXIT: 1
--- count build_index.py (expect >=2) ---
3
--- count agents/specialist.py (expect >=1) ---
1
```

- The two negative greps both return exit code 1 (no match found) — the
  shell-`!` negation in the spec passes when grep fails, so both
  assertions hold.
- `build_index.py` appears **3** times in the README (≥ 2 required): the
  three occurrences are (a) the Repository Structure tree, (b) the
  cardiology build command, (c) the endocrinology build command.
- `agents/specialist.py` appears **1** time (≥ 1 required): in the
  updated Architecture diagram line.

Additional sweep — `grep -nE "cardiologist\.py|endocrinologist\.py|build_cardio|build_endo" README.md`
returns exit 1 (no stale references in any form: bare filenames, paths,
or invocations).

## 4. Build Command Sanity Check

Per spec: *"a one-line statement confirming the build command works:
`python build_index.py --specialty cardiologist --help` exits 0"*

```
$ cd multi-agent_system
$ python build_index.py --specialty cardiologist --help
usage: build_index.py [-h] --specialty {cardiologist,endocrinologist}
                      [--chunk-size CHUNK_SIZE] [--keep-keywords]

Unified FAISS index builder. Reads specialty config from agents/registry.py
and builds the index for one specialty.

options:
  -h, --help            show this help message and exit
  --specialty {cardiologist,endocrinologist}
                        Specialty key from AGENT_REGISTRY (e.g. cardiologist,
                        endocrinologist).
  --chunk-size CHUNK_SIZE
                        Words per chunk (default: 400, the native size on
                        disk). Non-default values trigger re-chunking from the
                        existing native chunks; output goes to
                        data/processed/{specialty}_{chunk_size}_{keep|strip}/.
  --keep-keywords       Leave the `KEYWORDS:` header line inside
                        `page_content` instead of stripping it to metadata.
                        Used by the Stage 14 ablation.
EXIT: 0
```

**`python build_index.py --specialty cardiologist --help` exits 0.** The
unified builder is on disk, accepts the `--specialty` choices documented
in the refreshed README, and surfaces no import-time errors.

## 5. Files Touched

- `README.md` — three edit blocks, all in the Architecture / Repository
  Structure / Building sections; no narrative paragraphs altered
- `reports/report_stage_26.md` — this stage report (new)
