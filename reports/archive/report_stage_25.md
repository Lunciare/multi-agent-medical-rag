# Stage 25 — Adversarial Routing Test Set and Per-Category Evaluation

## 1. What Was Changed

Built a 32-case adversarial routing dataset and per-category evaluator to
stress-test the LLM router on inputs the clean-domain test split does not
exercise: typos, non-English queries, queries where the surface vocabulary
deliberately points to the wrong specialty, and pure-symptom queries
where either specialty is a defensible call.

- New `multi-agent_system/tests/data/adversarial_routing.json` — 32 cases,
  8 per category, all carrying `tier=4` / `tier_label="adversarial"` so
  they coexist with the existing tier 1/2/3 golden set without conflict.
- `multi-agent_system/tests/evaluate_routing.py`:
  - `SPLIT_TO_FILENAME` gained `"adversarial": "adversarial_routing.json"`
  - argparse `--split` choices expanded to include `"adversarial"`
  - Main loop now tracks `category_stats` alongside the existing
    `domain_stats` / `tier_stats`, and honours an optional `valid_domains`
    field on a case (used by `symptom_only_ambiguous`) — when present the
    case is correct if `predicted in valid_domains`; otherwise the old
    `predicted == expected_specialist` rule applies.
  - New stdout block and new markdown section
    `## Adversarial Routing — Per-Category Accuracy` are emitted only when
    `category_stats` is non-empty (i.e. only on `--split adversarial`); the
    `dev` / `test` / `all` splits are byte-unchanged in their output.

## 2. Per-Category Case Counts in `adversarial_routing.json`

Smoke-test command (verbatim from the spec):

```python
python -c "
import json
d = json.load(open('multi-agent_system/tests/data/adversarial_routing.json'))
assert len(d) >= 30, f'expected >=30 cases, got {len(d)}'
cats = set(c['category'] for c in d)
required = {'misspelled', 'non_english', 'dominant_pathology_mismatch',
            'symptom_only_ambiguous'}
assert cats == required, f'missing categories: {required - cats}'
per_cat = {c: sum(1 for x in d if x['category'] == c) for c in cats}
for c, n in per_cat.items():
    assert n >= 7, f'category {c} has only {n} cases (need >=7)'
print('Adversarial routing dataset OK:', per_cat)
"
```

Stdout:

```
Adversarial routing dataset OK: {'non_english': 8, 'symptom_only_ambiguous': 8, 'dominant_pathology_mismatch': 8, 'misspelled': 8}
```

| Category | Case count | Notes |
|---|---|---|
| `misspelled` | 8 | Cases like `adv_miss_1` ("atrail fibrlation", "sortness of breath"), `adv_miss_5` ("hyperthyroidsm", "undetctable"). 4 cardio + 4 endo. |
| `non_english` | 8 | Russian (4 cases), French (2), Spanish (2). 4 cardio + 4 endo. Russian queries cover both `cardiologist` (`adv_lang_1`, `adv_lang_4`) and `endocrinologist` (`adv_lang_5`, `adv_lang_8`) targets to test YandexGPT's native-language behaviour both ways. |
| `dominant_pathology_mismatch` | 8 | Surface vocabulary intentionally points to the opposite specialty from the actionable pathology — e.g. `adv_dom_1` foregrounds diabetes but the actionable issue is acute STEMI (expected: cardiologist); `adv_dom_3` foregrounds CAD history but the actionable issue is severe hypothyroidism (expected: endocrinologist). 3 cardio + 5 endo. |
| `symptom_only_ambiguous` | 8 | Each case carries `valid_domains: ["cardiologist", "endocrinologist"]` — symptoms only (fatigue, weight change, palpitations, dizziness, oedema). Treated as correct if the router picks either specialty. |

**Total: 32 cases** (≥ 30 required, ≥ 7 per category required, all 4
categories required — all checks pass).

## 3. Per-Category Accuracy from the Eval Run

Command (verbatim from the spec):

```
cd multi-agent_system
python tests/evaluate_routing.py --split adversarial
```

Eval output written to
`reports/routing_evaluation_2026-05-20_21-51-53.md`. Per-category block:

| Category | Correct | Total | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| `dominant_pathology_mismatch` | 8 | 8 | **100.0% [67.6%–100.0%]** |
| `misspelled` | 8 | 8 | **100.0% [67.6%–100.0%]** |
| `non_english` | 8 | 8 | **100.0% [67.6%–100.0%]** |
| `symptom_only_ambiguous` | 8 | 8 | **100.0% [67.6%–100.0%]** |
| **Overall (adversarial)** | **32** | **32** | **100.0% [89.3%–100.0%]** |

Per-case stdout marks every case `V` (correct). The single direction of
disagreement that would have been informative — a `symptom_only_ambiguous`
case where the LLM emitted something *outside* `valid_domains` — never
fired; every adversarial case produced a clean JSON `{"specialist": "..."}`
that parsed into either `cardiologist` or `endocrinologist`.

## 4. Direct Comparison — Clear-Domain Test vs Adversarial Per-Category

| Split | n | Correct | Accuracy [Wilson 95% CI] |
|---|---|---|---|
| **Clear-domain test set (§4.1, §4.8)** | 70 | 70 | **100.0% [94.8%–100.0%]** |
| Adversarial `misspelled` | 8 | 8 | 100.0% [67.6%–100.0%] |
| Adversarial `non_english` | 8 | 8 | 100.0% [67.6%–100.0%] |
| Adversarial `dominant_pathology_mismatch` | 8 | 8 | 100.0% [67.6%–100.0%] |
| Adversarial `symptom_only_ambiguous` | 8 | 8 | 100.0% [67.6%–100.0%] |
| **Adversarial overall** | **32** | **32** | **100.0% [89.3%–100.0%]** |

The point estimates match exactly (100.0% across the board). The Wilson
95% CIs differ only because the per-category n=8 is much smaller than the
test split n=70 — at n=8 a 100% observed rate is consistent with anything
from 67.6% upward; at n=70 the lower bound tightens to 94.8%. **The
adversarial set does not surface a measurable performance drop relative
to the clear-domain test set at this sample size**; doubling the per-category
n to 16 would tighten the lower bound to ~80% and at n=32/category to
~92%.

## 5. Categories Below 80% Accuracy

Per spec: *"any category where accuracy drops below 80% must be analysed
case-by-case."*

**No category is below 80% accuracy on this run.** All four categories
hit 8/8 (point estimate 100.0%; Wilson lower bound 67.6% by virtue of
the small n, not by virtue of any observed miss).

No per-category case-by-case analysis is therefore required by the spec
trigger. The full per-case `V`/`X` log is in
`reports/routing_evaluation_2026-05-20_21-51-53.md` for reference.

## 6. Notable Qualitative Observations (Not Triggered by Spec, But Worth Recording)

- **`dominant_pathology_mismatch` 8/8 is the most informative result.**
  These cases were authored specifically to fool a routing system that
  picks the specialty with the most matching vocabulary — e.g.
  `adv_dom_3` reads "70yo male with known CAD post-CABG ... fatigue, 9 kg
  weight gain, cold intolerance, bradycardia 46. TSH 78" — a Stage-11
  keyword router or a Stage-19 prompt without `domain_scope` could
  plausibly route to cardiologist on the CAD/CABG/bradycardia surface
  vocabulary. The fact that the Stage 24 prompt (which inlines each
  agent's `domain_scope` as a bullet block — see
  [report_stage_24.md](report_stage_24.md)) routes all 8 cases to the
  actionable specialty is consistent with the prompt change helping. We
  do not have a pre-Stage-24 baseline on adversarial cases, so this is a
  consistency observation, not a quantified claim.
- **`non_english` routes correctly in all three of Russian / French /
  Spanish.** YandexGPT is Russian-native and the 4 Russian cases were
  expected to work; the French (`adv_lang_2`, `adv_lang_6`) and Spanish
  (`adv_lang_3`, `adv_lang_7`) cases also routed correctly, suggesting
  the cognate medical vocabulary (`angine`, `hyperglycémie`, `Cushing`,
  `sus-décalage du segment ST`) carries enough signal for the router.
- **`symptom_only_ambiguous` distribution.** The router split the 8
  symptom-only queries 5 endocrinologist / 3 cardiologist — a more
  endo-leaning distribution than the 5 cardiologist / 3 endocrinologist
  split on the 8 `ambiguous_cases.json` queries (§4.2 table). This is
  consistent with stripping away the disease-naming vocabulary that
  anchors the clear-domain ambiguous cases, leaving general symptoms
  (fatigue, weight change, palpitations) that the LLM appears to map
  preferentially to endocrine differentials.

## 7. Files Touched

- `multi-agent_system/tests/data/adversarial_routing.json` — **new** (32 cases)
- `multi-agent_system/tests/evaluate_routing.py` — `SPLIT_TO_FILENAME`,
  argparse choices, per-category stats + valid_domains handling in the
  main loop, new stdout block, new markdown section
- `reports/report_final.md` — added subsection `### 4.2.1 Adversarial
  Routing` after §4.2 narrative (before §4.3 header) with the 4-row
  per-category table and a paragraph of interpretation
- `reports/routing_evaluation_2026-05-20_21-51-53.md` — side-effect file
  written by the adversarial eval run (per-category section is the new
  output; per-case detail table is the new sub-section)
- `reports/report_stage_25.md` — this stage report (new)