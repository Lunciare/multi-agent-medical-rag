# Corpus Inventory — 2026-05-21

Pre-build snapshot of the four specialist corpora that feed
`build_index.py`. Counts are derived purely from `data/processed/` —
no embedding/LLM API calls were made for this inventory. Cardiology
and endocrinology numbers are confirmed against the already-built
FAISS docstores (`vs.index.ntotal == 7,730` and `37,791` respectively,
matching the filesystem counts below to the chunk).

## Summary table

| Specialist | Documents | Chunks | Categories present | Mean chunk (words) | Max chunk (words) |
|---|---:|---:|---|---:|---:|
| **cardiology**         |   394 |  7,730 | Articles, Cases, Guidelines, Handbooks, Textbooks |  395.2 | 437 |
| **endocrinology**      | 1,156 | 37,791 | Articles, Cases, Guidelines, Textbooks            |  310.8 | 422 |
| **gastroenterologist** |   541 |  9,024 | Articles, Cases, Guidelines, Handbooks, Textbooks |  392.3 | 415 |
| **infection**          |   470 |  7,712 | Articles, Cases, Guidelines, Handbooks, Textbooks |  392.8 | 416 |
| **TOTAL**              | 2,561 | 62,257 | —                                                 |    —   |  —  |

Median chunk length is ≈402 words for every specialist (cardio 404,
endo 402, gastro 403, infect 402) — i.e. all four corpora were chunked
to the canonical `CHUNK_SIZE_WORDS = 400` setting in `settings.py`.
The lower mean for endocrinology is driven by a long tail of short
terminal chunks within the 27,342-chunk Textbooks bucket; the
distribution shape is unchanged across specialists.

## Per-category breakdown

`{N}d/{N}c` = `{N} documents / {N} chunks`. "—" = category absent.

| Specialist | Articles | Cases | Guidelines | Handbooks | Textbooks |
|---|---:|---:|---:|---:|---:|
| cardiology         | 7d / 10c    | 59d / 105c   | 113d / 1,478c | 190d / 972c | 25d / 5,165c  |
| endocrinology      | 67d / 2,487c | 74d / 1,046c | 101d / 6,916c | —           | 914d / 27,342c |
| gastroenterologist | 232d / 2,817c | 167d / 1,073c | 104d / 4,391c | 8d / 53c   | 30d / 690c    |
| infection          | 25d / 303c  | 59d / 390c   | 212d / 3,116c | 9d / 434c   | 165d / 3,469c |

Notes:
- Cardiology Articles is `.json` (10 single-text JSON records, treated as one
  virtual document per file by `build_index.py:_load_documents`). The other
  three specialists' Articles are conventional `0001.txt`/`0002.txt`/… chunks.
- Endocrinology has no Handbooks bucket.
- The gastroenterologist and infection corpora hit every category, with
  Guidelines as the largest single bucket (4,391 / 3,116 chunks respectively),
  not Textbooks — the opposite of the cardio/endo balance.

## KEYWORDS header convention

Sampled 50 chunks per specialist (random.seed=42). On all four corpora,
every sampled chunk contains a `KEYWORDS:` line, and on all four the
position is **line index 1** (i.e. the second line; line 0 is a short
document/section title). `build_index.py:_load_documents` strips the
`KEYWORDS:` line regardless of position (it iterates lines and
filters out any line starting with the prefix), so the new corpora's
position-1 placement is honoured correctly by the production loader.

Grep-confirmed coverage: 7,976 / 7,730 cardiology .txt chunks contain
the header, 38,947 / 37,791 endocrinology, 9,565 / 9,024 gastroenterologist,
8,181 / 7,712 infection — i.e. every chunk plus a small number of
`summary.txt` files (which the loader explicitly skips).

## Implications for downstream regression tests

- The four-specialist regression snapshot (Prompt 1 Part 5) will pick
  five canonical queries per new specialist out of corpora whose
  Guidelines bucket is the largest source of grounding evidence
  (4,391 chunks GI, 3,116 chunks ID) — query selection should bias
  toward common-condition language well-covered by those guideline
  documents.
- Chunk-size calibration that was tuned on 400-word cardio/endo chunks
  transfers cleanly: median word counts on the two new corpora are
  402–403 (cardio = 404, endo = 402), so any `MAX_L2_DISTANCE = 1.2`
  or `SIMILARITY_TOP_K = 5` hyper-parameters that were stable on cardio/endo
  should not be invalidated by a chunk-length distribution shift.
