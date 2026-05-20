# Stage 16 Report: Snapshot-Backed Retrieval Regression Test

**Date:** 2026-05-20

## 1. What Was Changed

- `multi-agent_system/tests/save_test_vectors.py`: extended end-to-end. After embedding the 10 canonical queries, the script now also runs each query against the correct domain's FAISS index (via the LangChain wrapper, exactly mirroring `Agent.answer`'s retrieval) and records the top-K=5 `source_file` list and L2 distances per query into `test_retrieval_snapshot.json`. The script gained a `--update-snapshot` flag: without it the snapshot is preserved (script refuses to overwrite), with it the snapshot is rewritten — the deliberate-rebuild ergonomic.
- `multi-agent_system/tests/data/test_retrieval_snapshot.json` (new): the canonical 10-query snapshot.
- `tests/test_retrieval_regression.py`: rewritten. The old test only asserted "≥1 chunk within `MAX_L2_DISTANCE=1.2`"; the new test additionally asserts (a) set-equality on the retrieved `source_file` list, and (b) per-rank L2 drift < 0.1 absolute. Both assertions fire with a verbose diff on failure (added / removed source files, per-rank distance deltas). The legacy sanity check is preserved as a second test (`test_every_query_returns_at_least_one_chunk`). The new test loads FAISS via raw `faiss.read_index` + the pickled docstore, bypassing the conftest's `MagicMock` over `langchain_community.vectorstores` (which would otherwise break unpickling).
- `README.md` "Running the Pytest Suite" section: expanded the `test_retrieval_regression.py` description to cover the new two-assertion behaviour; added a new **Regression testing** subsection documenting the `--update-snapshot` workflow.

## 2. Snapshot File Format

The snapshot is a JSON dict keyed by query string. Each value carries the top-K=5 retrieval shape:

```json
{
  "atrial fibrillation": {
    "top_k_source_files": ["0001.txt", "0002.txt", "0010.txt", "0003.txt", "0009.txt"],
    "top_k_l2_distances": [0.8394, 0.9068, 0.9108, 0.9223, 0.9280],
    "domain": "cardiology",
    "snapshot_date": "2026-05-20"
  }
}
```

L2 distances are squared L2 (LangChain's `IndexFlatL2` default; matches `MAX_L2_DISTANCE = 1.2` in `settings.py`).

## 3. Canonical 10-Query Snapshot (2026-05-20)

### Cardiology

| Query | Top-5 source_files (in retrieval order) | Top-5 L2 distances |
|---|---|---|
| atrial fibrillation | `0001.txt`, `0002.txt`, `0010.txt`, `0003.txt`, `0009.txt` | 0.8394, 0.9068, 0.9108, 0.9223, 0.9280 |
| hypertension management | `0001.txt`, `0023.txt`, `0014.txt`, `0019.txt`, `0001.txt` | 0.9599, 0.9629, 0.9629, 0.9634, 0.9682 |
| chest pain differential | `0006.txt`, `0004.txt`, `0003.txt`, `0002.txt`, `0002.txt` | 0.8354, 0.8391, 0.8563, 0.8606, 0.8639 |
| echocardiography | `0137.txt`, `0003.txt`, `0001.txt`, `0004.txt`, `0001.txt` | 0.8984, 0.9044, 0.9227, 0.9383, 0.9414 |
| beta blocker | `0006.txt`, `0024.txt`, `0002.txt`, `0023.txt`, `0001.txt` | 0.9672, 1.0216, 1.0546, 1.0811, 1.0881 |

### Endocrinology

| Query | Top-5 source_files (in retrieval order) | Top-5 L2 distances |
|---|---|---|
| type 2 diabetes | `0059.txt`, `0001.txt`, `0001.txt`, `0035.txt`, `0004.txt` | 0.8270, 0.8370, 0.8648, 0.8669, 0.8675 |
| thyroid nodule | `0009.txt`, `0005.txt`, `0001.txt`, `0010.txt`, `0005.txt` | 0.8179, 0.8374, 0.8407, 0.8462, 0.8517 |
| insulin resistance | `0002.txt`, `0001.txt`, `0004.txt`, `0005.txt`, `0017.txt` | 0.7921, 0.8010, 0.8460, 0.8646, 0.8711 |
| HbA1c | `0012.txt`, `0012.txt`, `0016.txt`, `0139.txt`, `0003.txt` | 0.8199, 0.8370, 0.8537, 0.8566, 0.8577 |
| adrenal insufficiency | `0001.txt`, `0002.txt`, `0013.txt`, `0005.txt`, `0005.txt` | 0.7707, 0.7867, 0.8269, 0.8292, 0.8313 |

(Duplicate `source_file` values across rows are expected — FAISS retrieves at chunk granularity, and several documents have multiple 400-word chunks ranked in the top-5 for the same query.)

## 4. Pytest Verification on Canonical Indices

```text
$ python -m pytest tests/test_retrieval_regression.py -v
============================= test session starts =============================
tests/test_retrieval_regression.py::test_retrieval_snapshot_matches PASSED
tests/test_retrieval_regression.py::test_every_query_returns_at_least_one_chunk PASSED
======================== 2 passed, 3 warnings in 0.24s ========================
```

## 5. Demonstration: Test Fails When Cardiology Index Is the Stage 14 `--keep-keywords` Build

The Stage 14 ablation produced `data/processed/cardiology_400_keep/faiss_index/` — same 7,730 chunks, same chunk size, but with the `KEYWORDS:` header line kept inside `page_content` so the embedder receives a different input. The regression test was re-run against this index (by swapping `data/processed/cardiology` → symlink to `cardiology_400_keep`, then restoring) and **all five cardiology queries fail with verbose diffs**:

```text
$ python -m pytest tests/test_retrieval_regression.py::test_retrieval_snapshot_matches
E   Failed: 5 regression failure(s):
E     query: 'atrial fibrillation' (domain=cardiology)
E       - removed (snapshot but not live):  ['0009.txt']
E       snapshot top-5: [('0001.txt', 0.8394), ('0002.txt', 0.9068), ('0010.txt', 0.9108), ('0003.txt', 0.9223), ('0009.txt', 0.9280)]
E       live     top-5: [('0001.txt', 0.8583), ('0002.txt', 0.8994), ('0003.txt', 0.9003), ('0010.txt', 0.9009), ('0001.txt', 0.9074)]
E     query: 'hypertension management' (domain=cardiology)
E       + added (live but not in snapshot): ['0010.txt']
E       - removed (snapshot but not live):  ['0023.txt']
E     query: 'chest pain differential' (domain=cardiology)
E       + added (live but not in snapshot): ['0001.txt', '0008.txt']
E       - removed (snapshot but not live):  ['0003.txt']
E     query: 'echocardiography' (domain=cardiology)
E       + added (live but not in snapshot): ['0005.txt']
E     query: 'beta blocker' (domain=cardiology)
E       + added (live but not in snapshot): ['0005.txt']
E       - removed (snapshot but not live):  ['0001.txt']
```

Per-query failure summary:

| Query | Failure mode (against `cardiology_400_keep`) |
|---|---|
| atrial fibrillation | Set mismatch: live retrieval drops `0009.txt` from the top-5; new ordering surfaces `0001.txt` twice. |
| hypertension management | Set mismatch: `0010.txt` enters top-5, `0023.txt` falls out. |
| chest pain differential | Set mismatch: `0001.txt` + `0008.txt` enter top-5, `0003.txt` falls out (largest delta seen). |
| echocardiography | Set mismatch: `0005.txt` enters top-5. |
| beta blocker | Set mismatch: `0005.txt` enters top-5, `0001.txt` falls out. |

This is the **intended behaviour**: keeping the `KEYWORDS:` line inside `page_content` shifts each chunk's embedding (the dense tokens dominate the vector), which permutes the top-5 ranking. The regression test catches the drift across the board and prints exactly which document IDs moved, so the operator knows whether the change was intentional. After confirming this *is* intentional, the operator would re-run `python tests/save_test_vectors.py --update-snapshot` against the new canonical index and commit the new snapshot file.

The 5 endocrinology queries continued to pass during this demonstration because the endocrinology index was untouched — confirming the test isolates failures by domain.

After restoring the original cardiology index, the test passes again:

```text
$ python -m pytest tests/test_retrieval_regression.py -v
======================== 2 passed, 3 warnings in 0.24s ========================
```

## 6. Smoke Test Output

```text
$ cd /tmp && rm -rf fake_repo && mkdir fake_repo && cd fake_repo
$ python -c "
import json
snap = {'atrial fibrillation': {'domain': 'cardiology', 'top_k_source_files': ['a.txt','b.txt','c.txt','d.txt','e.txt'], 'top_k_l2_distances': [0.8,0.9,1.0,1.1,1.15]}}
actual_files = ['e.txt','d.txt','c.txt','b.txt','a.txt']
actual_dists = [0.81,0.91,1.0,1.1,1.15]
key = 'atrial fibrillation'
assert set(actual_files) == set(snap[key]['top_k_source_files']), 'file set mismatch'
assert all(abs(a-s) < 0.1 for a,s in zip(sorted(actual_dists), sorted(snap[key]['top_k_l2_distances']))), 'distance drift'
print('Regression test smoke test passed')
"
Regression test smoke test passed
```

## 7. Workflow Summary (as documented in the new README subsection)

After **any** change to chunking, embeddings, or index parameters, regenerate the snapshot:

```bash
cd multi-agent_system
python tests/save_test_vectors.py --update-snapshot
```

Then commit the new `tests/data/test_retrieval_snapshot.json` alongside the change that caused it. Running `save_test_vectors.py` *without* `--update-snapshot` is a safe no-op on the snapshot file (it still refreshes `.npy` and `.json` if they are missing).

## 8. Open Questions

- **Squared-L2 labelling.** The codebase consistently labels distances as "L2" but stores squared L2 (the FAISS `IndexFlatL2` default that LangChain passes through verbatim). The snapshot's `top_k_l2_distances` field reuses the same loose label for consistency with `MAX_L2_DISTANCE = 1.2`. A future pass could rename throughout to `top_k_l2_squared` for precision; today's choice favours codebase-uniformity over mathematical strictness.
- **conftest mock bypass.** `tests/test_retrieval_regression.py` pops `langchain_*` modules from `sys.modules` before unpickling the FAISS docstore — the sibling tests' MagicMock would otherwise crash pickle. This couples this one test to that implementation detail; a future cleanup could move the regression test into its own `tests/integration/` subdirectory with no conftest.
- **0.1-pp drift tolerance.** Chosen heuristically. With Yandex's pre-normalised 256-d embeddings the typical query-to-doc squared-L2 is 0.7–1.1, so 0.1 is roughly 10–14 % of the signal scale. Tighter tolerance would catch more drift; looser would survive minor float-precision wobbles. A future ablation could sweep the tolerance against synthetic perturbations.

## 9. Commit Message Suggestion
`[tests] feat: snapshot-backed retrieval regression test (5 cardio + 5 endo queries × top-5 source_files + L2 dists); fails verbosely on the Stage 14 cardiology_400_keep index, passes on canonical`
