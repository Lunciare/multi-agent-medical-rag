# Stage 24 — Inline `domain_scope` Into the Routing System Prompt

(Filename note: the original task spec named this report `report_stage_21.md`,
but that filename is already used by the committed Landis & Koch
marginal-degeneracy report (`8e71155f`). After user confirmation, this
report uses the next sequential number `report_stage_24.md`. Stage 23
covered the orchestrator dict-access migration.)

## 1. What Was Changed

Stage 8 §6 ("Open Questions / Trade-offs") flagged `domain_scope` as a
registry field exported but not yet consumed by the routing prompt — at
that point the field would have shifted the LLM's routing output and the
Stage 8 acceptance criterion was a byte-stable 100/100 dev-split routing
accuracy. Stage 23 finished the deferred call-site migration (last open
item from Stage 8); this stage finishes the second open item by wiring
`domain_scope` into the prompt.

`_routing_system_prompt` in [`multi-agent_system/orchestrator.py`](../multi-agent_system/orchestrator.py)
now iterates `self.allowed_specialists`, reads each agent's
`agent.domain_scope` attribute (set on every `SpecialistAgent` per
[`agents/specialist.py:65`](../multi-agent_system/agents/specialist.py:65)),
and prepends a `key: scope` block to the existing JSON-output instruction.
The closing JSON instruction is unchanged, so:

- the `response_format={"type":"json_object"}` server-side enforcement
  documented in Stage 19 §2 keeps working,
- `_parse_router_output` continues to validate strictly against
  `self.allowed_specialists`, with no alias coercion (Stage 19 §1).

## 2. Verbatim New Routing Prompt (as rendered for the current 2-specialist registry)

Captured from `python -c "...; print(orch._routing_system_prompt())"` —
copied verbatim, including the two-space indentation on the scope lines:

```text
You are a medical orchestrator. Determine which specialist should handle the request. The available specialists and their domain scopes are:
  - 'cardiologist': cardiovascular disorders, including ischaemic heart disease, arrhythmias, heart failure, valvular disease, hypertension, and vascular disorders
  - 'endocrinologist': endocrine disorders, including thyroid disease, diabetes mellitus, adrenal disorders, pituitary conditions, parathyroid and calcium metabolism disorders, reproductive endocrinology, and metabolic bone diseases

Output a single JSON object with key `specialist` whose value is one of: 'cardiologist', 'endocrinologist'. Do not output any other text. Example: {"specialist": "cardiologist"}.
```

Both scope lines come straight from `AGENT_REGISTRY["…"]["domain_scope"]`
in [`agents/registry.py`](../multi-agent_system/agents/registry.py) — no
text in the prompt was hand-written for a specific specialty, so adding a
third entry to the registry automatically produces a third scope line.

## 3. Smoke-Test Output

Command (verbatim from the spec):

```python
python -c "
from orchestrator import MedicalOrchestrator
from settings import DEFAULT_KNOWLEDGE_BASE_DIR
orch = MedicalOrchestrator(DEFAULT_KNOWLEDGE_BASE_DIR)
prompt = orch._routing_system_prompt()
assert 'cardiovascular' in prompt.lower() or 'cardiac' in prompt.lower(), \
  'cardiologist domain_scope not in prompt'
assert 'endocrin' in prompt.lower() or 'hormone' in prompt.lower() or \
       'metabolic' in prompt.lower(), 'endocrinologist domain_scope not in prompt'
assert prompt.count('specialist') >= 2
print('domain_scope wiring smoke test OK')
print(prompt)
"
```

Stdout (trimmed to the assertion line):

```
domain_scope wiring smoke test OK
```

(All three asserts passed; the rendered prompt that follows the OK line
is identical to §2 above.)

## 4. Pre / Post Dev Routing Accuracy

| Run | Prompt | Cardiology | Endocrinology | **Overall** | Wilson 95% CI |
|---|---|---|---|---|---|
| Pre-Stage-24 (Stage 19 wording) | JSON output, no scopes | 15/15 = 100.0% | 15/15 = 100.0% | **30/30 = 100.0%** | [88.6%–100.0%] |
| Post-Stage-24 | JSON output, with `domain_scope` block | 15/15 = 100.0% | 15/15 = 100.0% | **30/30 = 100.0%** | [88.6%–100.0%] |

Per-tier:

| Domain | Tier | Label | Correct/Total | Pre | Post |
|---|---|---|---|---|---|
| cardiologist | 1 | core | 14/14 | 100.0% | 100.0% |
| cardiologist | 3 | out_of_scope | 1/1 | 100.0% | 100.0% |
| endocrinologist | 1 | core | 15/15 | 100.0% | 100.0% |

`diff` between the two stdout captures (after stripping timestamp lines)
returns exit 0 — the outputs are byte-identical.

## 5. Per-Ambiguous-Case Routing Decisions (Pre vs Post)

The 8 cross-domain ambiguous cases are inspected separately because they
do not contribute to the accuracy count (`valid_domains` lists both
specialists) — they are the most sensitive probe of whether the prompt
change shifted the LLM's preferences.

| Case ID | Pathology hint | Pre (Stage 19 prompt) | Post (Stage 24 prompt) | Flipped? |
|---|---|---|---|---|
| ambig_1 | diabetic cardiomyopathy | `cardiologist` | `cardiologist` | no |
| ambig_2 | thyroid-induced atrial fibrillation | `endocrinologist` | `endocrinologist` | no |
| ambig_3 | SGLT2 inhibitor cardioprotection in ACS | `cardiologist` | `cardiologist` | no |
| ambig_4 | hyperaldosteronism with cardiac complications | `endocrinologist` | `endocrinologist` | no |
| ambig_5 | catecholamine-induced cardiomyopathy | `endocrinologist` | `endocrinologist` | no |
| ambig_6 | amiodarone-induced thyroid dysfunction | `endocrinologist` | `endocrinologist` | no |
| ambig_7 | metabolic syndrome with coronary artery disease | `cardiologist` | `cardiologist` | no |
| ambig_8 | carcinoid heart disease | `cardiologist` | `cardiologist` | no |

**Zero ambiguous cases flipped.** All 8 routing decisions are
character-identical between the pre- and post-Stage-24 runs. The
distribution (5 cardiologist / 3 endocrinologist) is preserved exactly.

## 6. Direct Statement: Revert Trigger

Per spec: *"If any case flips, revert and report."*

**No case flipped — neither the 30 deterministic dev cases nor any of
the 8 ambiguous cases.** The new prompt is therefore **NOT reverted**;
the Stage 24 patch to `_routing_system_prompt` stands. If you re-run the
dev eval at a different time and observe any flip, the revert path is to
restore the Stage 19 body of `_routing_system_prompt` (a single function
body — see the diff in §7 below).

## 7. Files Touched

- [`multi-agent_system/orchestrator.py`](../multi-agent_system/orchestrator.py) — `_routing_system_prompt` body replaced; the Stage 19 docstring was removed (the function body is now self-explanatory and the Stage history lives in this report)
- [`reports/report_stage_19.md`](report_stage_19.md) — §3 carries a Stage 24 update note pointing here
- `reports/report_stage_24.md` — this stage report (new)
- `reports/routing_evaluation_2026-05-20_21-41-49.md` — side-effect file written by the post-change `evaluate_routing.py` run (safe to delete or commit alongside)
