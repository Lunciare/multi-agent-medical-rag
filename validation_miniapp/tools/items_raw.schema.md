# Raw-inputs schema (`items_raw.json`) — team-filled

The team fills this file with the **real outputs** of their systems. `tools/build_items.py`
converts it into the `items.json` the app serves. The build tool never authors, corrects, or
edits the substance of any case or answer — your inputs here are **ground truth**.

> **This file contains study content + the RAG/vanilla mapping. It is git-ignored** (see the
> repo `.gitignore`). Keep the dummy fixture under `tests/fixtures/` for testing; put the real
> file at `tools/items_raw.json` (or pass `--raw <path>`).

## Format

A JSON array. One object per case. Common fields (always required):

| Field | Type | Notes |
|-------|------|-------|
| `item_id` | string | Unique within the file. Stable join key (DB rows, position map, CSV). |
| `gold_specialty` | `"cardiology"` \| `"endocrinology"` | Ground-truth specialty for the case. |
| `available_specialties` | list of strings | Specialists the orchestrator could route to. Non-empty. |
| `routed_specialty` | string | What the **real pipeline run** chose. Must be one of `available_specialties`. |

Then **one of two content variants**, chosen by build `--mode`:

### Variant A — manual (RU already written) — `--mode manual`
You supply Russian directly; the build does **no** machine translation (passthrough).

| Field | Type | Notes |
|-------|------|-------|
| `case_ru` | string | The case text, in Russian. |
| `answer_rag_ru` | string | **Your RAG pipeline's** answer, in Russian. RAG arm. |
| `answer_vanilla_ru` | string | **Your vanilla LLM's** answer, in Russian. Vanilla arm. |

### Variant B — assisted (English in, machine-translated) — `--mode assisted`
You supply English; the build calls a translator you configure (`--translator`) and marks
every output **DRAFT — needs human review**.

| Field | Type | Notes |
|-------|------|-------|
| `case_en` | string | The case text, in English. |
| `answer_rag_en` | string | **Your RAG pipeline's** answer, in English. RAG arm. |
| `answer_vanilla_en` | string | **Your vanilla LLM's** answer, in English. Vanilla arm. |

## Hard rules (integrity)

1. `answer_rag_*` must contain **only** your RAG system's output; `answer_vanilla_*` must
   contain **only** your vanilla system's output. **Never swap or mix them** — a swap silently
   inverts the entire study. The build copies these fields straight through, arm-labelled.
2. `routed_specialty` is whatever your real pipeline chose — even if it equals/differs from
   `gold_specialty`. Do not "fix" it.
3. Do not pre-clean the answers to hide which system produced them; the build's normalization
   step removes **tells only** (citation markers, source headers, arm boilerplate) identically
   from both arms, and emits a diff for human review. Author nothing.
4. Both arms are translated and normalized by the **identical** method/settings, so style and
   formatting cannot leak the arm.

## Example (manual variant)

```json
[
  {
    "item_id": "case-001",
    "gold_specialty": "cardiology",
    "available_specialties": ["cardiology", "endocrinology"],
    "routed_specialty": "cardiology",
    "case_ru": "…",
    "answer_rag_ru": "…",
    "answer_vanilla_ru": "…"
  }
]
```

For the assisted variant, replace the three `*_ru` fields with `case_en`, `answer_rag_en`,
`answer_vanilla_en`.
