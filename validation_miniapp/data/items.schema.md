# Item schema (`items.*.json`)

The items file is a JSON array. Each element is one **case** shown to a rater for a
blinded pairwise comparison. The backend loads this file at startup and never mutates it
(it is read-only reference data). Rater judgments are stored separately in SQLite.

> The shipped `items.dummy.json` contains **fake, non-medical placeholder text**, clearly
> marked with `[ЗАГЛУШКА ...]`. It exists only so the app and smoke test run end-to-end
> without any real clinical content. Replace it with real RU cases in a later stage.

## Fields

| Field                   | Type            | Required | Description |
|-------------------------|-----------------|----------|-------------|
| `item_id`               | string          | yes      | Stable unique id for the case. Used as a join key everywhere (DB rows, position map, CSV). Must be unique within the file. |
| `case_ru`               | string          | yes      | The clinical case / question text shown to the rater, in Russian. |
| `gold_specialty`        | enum            | yes      | Ground-truth specialty for this case. One of the four canonical values: `cardiology` \| `endocrinology` \| `gastroenterology` \| `infectious_diseases`. Used to score routing objectively in export. |
| `routed_specialty`      | enum            | yes      | The (canonical) specialty the orchestrator actually chose for this case. Normally one of `available_specialties`. Shown to the rater on the routing screen (in Russian — see below). May differ from `gold_specialty` (that is what we measure). |
| `available_specialties` | list of strings | yes      | The set of specialists the orchestrator could have routed to — the real four, in canonical form. Shown as context on the routing screen. |

### Canonical specialties + Russian display

Stored values use one canonical vocabulary (see `backend/specialties.py`):
`cardiology`, `endocrinology`, `gastroenterology`, `infectious_diseases`. Both the dataset
labels and the orchestrator's emitted routing labels map to these. The rater-facing routing
screen displays them in Russian (display only; stored data stays canonical):

| canonical | Russian |
|-----------|---------|
| `cardiology` | Кардиолог |
| `endocrinology` | Эндокринолог |
| `gastroenterology` | Гастроэнтеролог |
| `infectious_diseases` | Инфекционист |
| `answer_rag_ru`         | string          | yes      | Answer produced by the **full pipeline** (orchestrator routing + retrieval). Russian. This is the "RAG" arm. Never labelled as such to the rater. |
| `answer_vanilla_ru`     | string          | yes      | Answer produced by the **vanilla LLM** (no routing, no retrieval). Russian. This is the "vanilla" arm. Never labelled as such to the rater. |

## Arm semantics (important for de-blinding)

There are exactly two **arms** per item:

- `rag`     → text in `answer_rag_ru`
- `vanilla` → text in `answer_vanilla_ru`

The rater never sees these labels. The server randomly maps each arm to a neutral display
position (`option_1` / `option_2`) per `(rater, item)` and persists that map. Export uses
the stored map to convert the rater's positional preference back into a signed
preference toward the RAG arm. See `backend/db.py` and `backend/export_csv.py`.

## Constraints / conventions

- `item_id` unique within the file.
- `routed_specialty` should be an element of `available_specialties`.
- `gold_specialty` should be an element of `available_specialties`.
- All `*_ru` text is plain UTF-8; no HTML. The frontend renders it as text, not markup.
- The two answers should be roughly comparable in length/format so length is not an
  obvious blinding tell (placeholder data deliberately varies to exercise the UI).
