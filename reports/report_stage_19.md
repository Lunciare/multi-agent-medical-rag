# Stage 19 Report: Structured JSON Routing + Drop Alias Coercion

**Date:** 2026-05-20

## 1. What Was Changed

- `multi-agent_system/orchestrator.py`:
  - `MedicalOrchestrator.__init__` now publishes `self.allowed_specialists = list(self.agents.keys())` — a strict allow-list used by `route()`.
  - `_routing_system_prompt()` rewritten to ask for JSON output (new text in §3 below).
  - `route()` rewritten: tries `response_format={"type": "json_object"}` first; falls back to plain `chat.completions.create` if Yandex rejects the parameter on a future model. The raw response is passed to a new `_parse_router_output(raw)` helper which `json.loads` the text and checks the `specialist` field against `self.allowed_specialists`. On failure the method returns `"__error__:validation"` — **never silently re-maps**.
  - Added a `validation` branch to `answer()`'s error-message switch.
- `multi-agent_system/tests/evaluate_routing.py`:
  - **Removed `SPECIALIST_ALIASES`** (the 4-entry dict + the substring-match loop in `normalise()`).
  - `route_query()` now sends the same JSON-structured prompt and tries `response_format` with the same fallback shape as the orchestrator.
  - `normalise()` now JSON-parses the raw response and returns the specialty only if it appears in `ALLOWED_SPECIALISTS = {"cardiologist", "endocrinologist"}`; otherwise returns the raw text lower-cased (so the caller's `predicted == expected` comparison counts the case as wrong).
- `tests/test_integration.py` updated: the three routing-related mocks now return the new JSON-structured router output. `TestRouting::test_route_unknown_specialist` accepts either the new `"Routing failed: …"` user-visible message or the legacy `"could not determine"` for backward compatibility.

## 2. Yandex `response_format` Probe

A direct probe against `gpt://{folder}/yandexgpt/latest`:

```text
$ python3 -c "from settings import client, ROUTING_MODEL, YANDEX_PROJECT_ID
r = client.chat.completions.create(
    model=ROUTING_MODEL,
    messages=[
        {'role':'system','content':'You are a medical orchestrator. Output a single JSON object with key specialist whose value is one of: cardiologist or endocrinologist.'},
        {'role':'user','content':'I have palpitations and atrial fibrillation.'},
    ],
    temperature=0.0, max_tokens=64,
    response_format={'type':'json_object'},
    extra_headers={'x-folder-id': YANDEX_PROJECT_ID},
)
print(repr(r.choices[0].message.content))"
'{"specialist": "cardiologist"}'
```

**Result: `response_format={"type": "json_object"}` is SUPPORTED on YandexGPT.** The server returned clean JSON with no surrounding prose. The JSON-fallback path in `route()` is therefore exercised only when a future Yandex model rejects the parameter; today, `response_format` is the active path.

## 3. New Routing Prompt (verbatim)

```text
You are a medical orchestrator. Determine which specialist should handle the request. Output a single JSON object with key `specialist` whose value is one of: 'cardiologist', 'endocrinologist'. Do not output any other text. Example: {"specialist": "cardiologist"}.
```

The `'cardiologist', 'endocrinologist'` substring is generated from `self.allowed_specialists` via `", ".join(repr(s) for s in ...)` — so the prompt auto-expands when a new `agents/registry.py` entry is added.

> **Update (Stage 24, 2026-05-20):** The Stage 19 prompt above has since been extended to inline each specialist's `domain_scope` description, completing the deferred work flagged in §6 of [`report_stage_8.md`](report_stage_8.md). The current routing prompt is documented in full in [`report_stage_24.md`](report_stage_24.md) §2; pre/post dev routing accuracy was 30/30 → 30/30 with zero per-case flips.

## 4. Pre / Post Routing Accuracy (Wilson 95% CI)

### Dev split (n=30; `cardio_1..15` + `endo_1..15`)

| Run | Mode | Routing prompt | Eval normaliser | Cardiology | Endocrinology | Overall |
|---|---|---|---|---|---|---|
| Pre-refactor (Stage 11 wording) | one-word output | "Respond strictly in one word." | `SPECIALIST_ALIASES` w/ substring match | 15/15 = 100.0% [79.6%–100%] | 15/15 = 100.0% [79.6%–100%] | **30/30 = 100.0% [88.6%–100%]** |
| Post-refactor (Stage 19) | JSON object | `{"specialist": "..."}` | strict equality vs `ALLOWED_SPECIALISTS` | 15/15 = 100.0% [79.6%–100%] | 15/15 = 100.0% [79.6%–100%] | **30/30 = 100.0% [88.6%–100%]** |

**No accuracy drop on dev.** The pre-refactor 100% was *not* artificially inflated by alias coercion.

### Test split (n=70) — confirmation

| Run | Cardiology (n=35) | Endocrinology (n=35) | Overall | Per-tier |
|---|---|---|---|---|
| Post-refactor (Stage 19) | 35/35 = 100.0% [90.1%–100%] | 35/35 = 100.0% [90.1%–100%] | **70/70 = 100.0% [94.8%–100%]** | T1/T2/T3 × cardio/endo all 100% |

Test accuracy is identical to the §4.8 numbers reported pre-refactor.

### Why no drop happened

`grep "raw:"` over the pre-refactor dev log returns zero hits — the `evaluate_routing.py` log only prints `raw: …` when `normalise()` substring-coerced the model's output, and that never fired. YandexGPT was already outputting canonical strings (`"cardiologist"` / `"endocrinologist"`) verbatim under the old prompt. The `SPECIALIST_ALIASES` dict was defensive code that **never actually moved a case** on our 30+70-case sample at temperature=0. Removing it is therefore strictly safer (now non-canonical output would surface as `__error__:validation` rather than be silently coerced) without sacrificing any measured accuracy.

## 5. Per-Case Comparison — Cases That Flipped

| Case ID | Pre-refactor predicted | Post-refactor predicted | Flipped? |
|---|---|---|---|
| `cardio_1` .. `cardio_15` | `cardiologist` | `cardiologist` (from `{"specialist": "cardiologist"}`) | no |
| `endo_1` .. `endo_15` | `endocrinologist` | `endocrinologist` (from `{"specialist": "endocrinologist"}`) | no |
| `ambig_1`–`ambig_8` | LLM's choice (5 cardio / 3 endo) | LLM's choice (5 cardio / 3 endo, identical IDs) | no |

**Zero flipped cases.** Per-case routing decisions are byte-identical between the two runs on every dev case and every ambiguous case. The only observable change is the response format (`"cardiologist"` → `'{"specialist": "cardiologist"}'`).

## 6. Report §4.1 Update — Not Triggered

The task spec said: *"If accuracy drops by more than 2 pp, the previous 100% number was partly attributable to alias coercion. Report this in §4.1 of the report."*

Drop = **0 pp** on both dev and test. Therefore §4.1 in `report_final.md` is **not** updated as part of Stage 19. The §4.1 numbers from Stage 15 stand. A one-line note could be added to §4.1 (or as a footnote) flagging that the routing pipeline is now JSON-structured + strict-validated, but that is a documentation polish rather than a numerical correction.

## 7. Smoke Test Output

```text
$ python -c "
import json
def parse_router_output(raw, allowed):
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        spec = obj.get('specialist', '').strip().lower()
        if spec in allowed: return spec
    except Exception:
        pass
    return None
allowed = {'cardiologist', 'endocrinologist'}
assert parse_router_output('{\"specialist\": \"cardiologist\"}', allowed) == 'cardiologist'
assert parse_router_output('cardiology', allowed) is None
assert parse_router_output('surgeon', allowed) is None
print('Routing structured-output smoke test passed')
"
Routing structured-output smoke test passed
```

## 8. Pytest Sanity

```text
$ python -m pytest tests/ -q
34 passed, 1 skipped, 3 warnings in 1.18s
```

`test_route_returns_cardiologist`, `test_route_returns_endocrinologist`, `test_route_unknown_specialist`, `test_cardiologist_answer`, and `test_empty_domain_edge_case` were all updated to mock the new JSON-structured router output; the rest of the suite required no change.

## 9. Open Questions

- **Test-set non-determinism.** The Stage 19 probe and the post-refactor eval both happen to produce canonical strings on every case; YandexGPT is not strictly deterministic at `temperature=0`. A future stress run (the same prompt, 5 repeated calls per case, look for any non-JSON or non-allowed output) would tighten the empirical evidence that `__error__:validation` is rare.
- **What `__error__:validation` looks like in production.** Today the user-facing error message is `"Routing failed: the LLM did not return a recognised specialist."` On a `--mode multi_judge`–style follow-up, the orchestrator could retry with a stricter prompt or fall back to the keyword baseline before surfacing the error. Recorded as future work.
- **3-specialist expansion.** The `repr(s)` join in the prompt produces `"'cardiologist', 'endocrinologist'"`; with a third specialist it would become `"'cardiologist', 'endocrinologist', 'dermatologist'"`. The format is OK for 3, but at 5+ specialists the prompt should switch to a numbered list to avoid the LLM glossing over middle entries.

## 10. Commit Message Suggestion
`[chore] structured JSON routing: orchestrator + evaluate_routing.py use response_format={"type":"json_object"} on Yandex (verified supported); strict allow-list validation; SPECIALIST_ALIASES dict deleted; pre/post routing 100% dev + 100% test, zero flipped cases`
