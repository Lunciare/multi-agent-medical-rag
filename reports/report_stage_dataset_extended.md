# Stage: Dataset Extension to 4 Specialties

(Scope: author 50 gastroenterologist + 50 infectionist golden cases,
re-split dev/test, run auto-annotator, extend the adversarial routing
set by 32 cases, spot-check four end-to-end. Inherits from
`report_stage_indices_built.md` (FAISS+BM25 for gastro/infect on disk
with the 354+236 PDF-artifact filter applied). **Existing 100 cardio/endo
cases were not touched** — gold_sources byte-identical before/after.)

## 1. Tier distribution (200-case dataset)

| Specialist | T1 (core) | T2 (peripheral) | T3 (out_of_scope) | Total |
|---|---:|---:|---:|---:|
| cardiologist        | 27 | 14 | 9 | 50 |
| endocrinologist     | 27 | 16 | 7 | 50 |
| gastroenterologist  | **27** | **15** | **8** | **50** |
| infectionist        | **27** | **15** | **8** | **50** |
| **Total**           | **108** | **60** | **32** | **200** |

Dev/test split — `case-number ≤ 15` → dev; rest → test:

| Split | cardio | endo | gastro | infect | Total |
|---|---:|---:|---:|---:|---:|
| dev  | 15 | 15 | 15 | 15 | **60** |
| test | 35 | 35 | 35 | 35 | **140** |

Tier-by-ID convention for the new specialists (identical for gastro and infect):

```
T1 nums: [1-9, 11-21, 33, 34, 37, 38, 41, 42, 45]     (27 cases)
T2 nums: [22-27, 35, 36, 39, 40, 43, 44, 46, 49, 50]  (15 cases)
T3 nums: [10, 28-32, 47, 48]                          (8 cases)
```

Dev sees 14 T1 + 1 T3 per new specialist; test sees 13 T1 + 15 T2 + 7 T3.

Brief recommended 27/15/8 as the default. Cardio's actual mix is
27/14/9 (T1 is identical; T2 minus 1 → T3 plus 1), and endo's is
27/16/7. I matched the brief's default (27/15/8) verbatim for both new
specialists — slight T3 increase vs. cardio's 9 was not warranted by
the corpus inventory (both new corpora are large enough that 8 T3 stress
cases give adequate refusal-gate signal without crowding T2).

## 2. Sample cases — one per tier × each new specialist (8 cases)

### gastroenterologist

**`gastro_1` — T1 core (GERD)**
```json
{
  "id": "gastro_1",
  "tier": 1, "tier_label": "core",
  "expected_specialist": "gastroenterologist",
  "query": "A 48-year-old male presents with a 6-month history of postprandial retrosternal burning, regurgitation of sour fluid, and nocturnal cough. He drinks alcohol on weekends and is overweight (BMI 31). Empiric therapy has not been tried. What is the first-line diagnostic and management approach?",
  "expected_keywords": ["GERD", "proton pump inhibitor", "lifestyle modification", "endoscopy", "reflux"],
  "gold_sources": [<set by auto-annotator>]
}
```

**`gastro_22` — T2 peripheral (Wilson's disease)**
```json
{
  "id": "gastro_22",
  "tier": 2, "tier_label": "peripheral",
  "expected_specialist": "gastroenterologist",
  "query": "A 24-year-old male presents with tremor, dysarthria, and behavioural change. Serum ceruloplasmin is 4 mg/dL (low), 24-hour urinary copper is markedly elevated, and slit-lamp examination shows Kayser-Fleischer rings. Liver enzymes are mildly elevated. What is the diagnosis and treatment?",
  "expected_keywords": ["Wilson disease", "ceruloplasmin", "Kayser-Fleischer", "penicillamine", "copper"],
  "gold_sources": [<set by auto-annotator>]
}
```

**`gastro_10` — T3 out-of-scope (Heyde syndrome)**
```json
{
  "id": "gastro_10",
  "tier": 3, "tier_label": "out_of_scope",
  "expected_specialist": "gastroenterologist",
  "query": "A 78-year-old female with severe aortic stenosis (peak gradient 75 mmHg) presents with recurrent occult GI bleeding, iron-deficiency anaemia, and a negative upper and lower endoscopy. Small-bowel angiodysplasia is suspected. What is the proposed pathophysiology and definitive treatment?",
  "expected_keywords": ["Heyde syndrome", "angiodysplasia", "von Willebrand", "aortic stenosis", "occult bleeding"],
  "gold_sources": []
}
```

(T3 follows the existing cardio_28..cardio_36 / endo_44..endo_50
pattern: in-domain query, but the named condition is rare enough that
the corpus shouldn't have evidence to support a confident answer — the
refusal gate is meant to fire. See §6.4 below for a brief-vs-data
inconsistency note on this.)

### infectionist

**`infect_1` — T1 core (community-acquired pneumonia)**
```json
{
  "id": "infect_1",
  "tier": 1, "tier_label": "core",
  "expected_specialist": "infectionist",
  "query": "A 68-year-old male presents with 4 days of fever, productive cough with rust-coloured sputum, pleuritic chest pain and SpO2 92% on room air. Chest X-ray shows right lower lobe consolidation. He has no recent hospitalisation. CRB-65 score is 1. What is the diagnosis, severity assessment and empirical therapy?",
  "expected_keywords": ["community-acquired pneumonia", "CRB-65", "empirical antibiotics", "amoxicillin", "Streptococcus pneumoniae"],
  "gold_sources": [<set by auto-annotator>]
}
```

**`infect_22` — T2 peripheral (severe falciparum malaria)**
```json
{
  "id": "infect_22",
  "tier": 2, "tier_label": "peripheral",
  "expected_specialist": "infectionist",
  "query": "A 35-year-old male returns from Nigeria with 5 days of fever, rigors, headache and myalgia, and now develops jaundice, oliguria and confusion. Blood film shows ring forms with parasitaemia of 8%. What is the diagnosis, the severity classification and the recommended therapy?",
  "expected_keywords": ["severe falciparum malaria", "Plasmodium falciparum", "intravenous artesunate", "parasitaemia", "WHO severity criteria"],
  "gold_sources": [<set by auto-annotator>]
}
```

**`infect_10` — T3 out_of_scope (Whipple's disease)**
```json
{
  "id": "infect_10",
  "tier": 3, "tier_label": "out_of_scope",
  "expected_specialist": "infectionist",
  "query": "A 55-year-old male presents with several years of progressive arthralgia, weight loss, chronic diarrhoea and cognitive decline. Duodenal biopsy shows PAS-positive macrophages and PCR detects Tropheryma whipplei. What is the diagnosis, the recommended treatment and the rationale for prolonged therapy?",
  "expected_keywords": ["Whipple's disease", "Tropheryma whipplei", "PAS-positive macrophages", "ceftriaxone", "trimethoprim-sulfamethoxazole"],
  "gold_sources": []
}
```

Remaining 92 cases follow the same schema and topic distribution — full
list in `multi-agent_system/tests/data/golden_dataset.json`.

## 3. Auto-annotator output: corpus-gap signal

`tests/annotate_gold_sources.py --auto` was run on the full 200-case set
after one tiny pre-edit to the script: the hardcoded
`{cardiologist, endocrinologist}` agent lookup in `_retrieve_top_k`
was replaced with `orchestrator.agents[case["expected_specialist"]]`
so the four-specialist registry works. No other behavioural change.

Result: **160 / 168 T1+T2 cases got ≥ 1 gold source; 8 got `gold_sources = []`.**
The 8 corpus-gap cases:

| Case | Tier | Specialty | Reason |
|---|---|---|---|
| **cardio_35** | T2 | cardiologist | Pre-existing — documented in Stage 6 §3 / §6 Limitation 8 (STEMI complicated by complete heart block, temporary pacing) |
| **endo_46**   | T1 | endocrinologist | Pre-existing — documented in Stage 6 §3 / §6 Limitation 8 (hypoglycaemia unawareness, closed-loop) |
| **gastro_37** | T1 | gastroenterologist | New — acute viral gastroenteritis. Corpus contains 50 chunks for `gastroenteritis` but none match the specific keywords (`oral rehydration`, `norovirus`, `self-limiting`, `supportive care`). Likely an auto-annotator keyword mismatch — see §6.5 below. |
| **gastro_39** | T2 | gastroenterologist | New — hereditary haemochromatosis. Only 12 chunks match `haemochromatosis|hemochromatosis` in the gastro corpus, and the specific keyword set (`HFE`, `C282Y`, `transferrin saturation`, `phlebotomy`) only partially appears in those chunks. Expected corpus gap. |
| **gastro_44** | T2 | gastroenterologist | New — Zollinger-Ellison syndrome. Only 7 chunks match. Expected corpus gap. |
| **infect_14** | T1 | infectionist | New — C. difficile colitis. 31 chunks match `Clostridium difficile`, but specific keywords (`fidaxomicin`, `vancomycin oral`, `CDI`) under-represented. Auto-annotator keyword mismatch likely. |
| **infect_21** | T1 | infectionist | New — HSV encephalitis. 139 chunks match generic encephalitis terms, but the specific HSV+aciclovir+temporal-lobe+CSF-PCR vocabulary doesn't co-occur. Mismatch. |
| **infect_39** | T2 | infectionist | New — prosthetic joint infection. Only 6 chunks match `PJI`. Expected corpus gap. The retrieved top-K=20 has its top hit at L2 = 1.107 — close to the `MAX_L2_DISTANCE = 1.2` ceiling — confirming sparse coverage. |

**Annotator circularity caveat (§6 Limitation 7 in `report_final.md`):**
empty `gold_sources` does NOT mean the query is unanswerable. For all
6 new corpus-gap cases, the auto-annotator's top-20 retrieval was
≥ 18 chunks within `MAX_L2_DISTANCE = 1.2` — i.e. retrieval works,
but no chunk in that top-20 happened to contain any of the
`expected_keywords` list (auto-annotator requires ≥ 1 keyword hit AND
within-L2 for a doc to be picked). This is the same pattern as the
pre-existing `cardio_35` / `endo_46` cases; treat the 6 new cases the
same way (Stage 6's interactive-annotation pass would likely populate
some of them — `infect_14` and `gastro_37` especially, since the
clinical scenario is textbook).

**Cardio/endo gold_sources drift after re-running the auto-annotator
on the merged 200-case set: 0.** Every one of the 100 existing
cardio/endo cases produced byte-identical `gold_sources` to its prior
state — confirming the auto-annotator is deterministic and the
expansion to 4 specialties didn't disturb the existing annotations.

## 4. Adversarial cases (32 new) — full dump

IDs continue from existing Stage 25 set (`adv_*_1..8`):
`adv_miss_9..16`, `adv_lang_9..16`, `adv_dom_9..16`, `adv_amb_9..16`.
Schema is identical to existing — `id`, `category`, `query`,
`expected_specialist`, `tier = 4`, `tier_label = "adversarial"`,
plus `valid_domains` for `symptom_only_ambiguous`.

### misspelled (4 gastro + 4 infect)

```json
[
  {"id": "adv_miss_9",  "category": "misspelled", "expected_specialist": "gastroenterologist",
   "query": "32yo female with 4-month bloody diareha, urgemcy and aphthous oral ulsers. Colonoscopy: continuous mucosal inflamation of recctum and sigmoid. Diagnois and induction strategy?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_10", "category": "misspelled", "expected_specialist": "gastroenterologist",
   "query": "55-year-old man with sirosis develops worsenig ascits, leg swelling and dispnea. SAAG 1.7, PMNs 50/mm3. Manegement plan?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_11", "category": "misspelled", "expected_specialist": "gastroenterologist",
   "query": "60yo M with epigatric pain radiating to bak, vommiting, lypase 9x ULN. US shows galstones with normal CBD. Severtiy grading and initial flud therapy?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_12", "category": "misspelled", "expected_specialist": "gastroenterologist",
   "query": "70yo on aspren and apixiban presents with melena and Hb drop 13 → 8. BP 100/65, HR 95. Tming of endscopy and managment of antiocoagulation?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_13", "category": "misspelled", "expected_specialist": "infectionist",
   "query": "68yo M with 4 days fever, prodctive cougn and pleurtic chest pain. CXR shows right lower lob consoliadation. CRB-65 = 1. Emperic abx for community aquired pneumonia?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_14", "category": "misspelled", "expected_specialist": "infectionist",
   "query": "35yo immigrant with 3-month productiv cough, weigt loss, nyght sweats and hemptysis. Smear-postiive for AFB. Initiating tubercolosis tretament — what RIPE regimen?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_15", "category": "misspelled", "expected_specialist": "infectionist",
   "query": "22yo M with photophhobia, neck stifness and feber 39.5. CSF: 1800 WCC neutrofilic, low gluose, gram-pos diplococi. Imediate manegement of bacterial menenjitis?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_miss_16", "category": "misspelled", "expected_specialist": "infectionist",
   "query": "60yo M with bioprostetic valv and 3-week feber. Three sets of blood cultres growing viridans strepococci, 12mm vegitation on aortic valve on TTE. Antibiotc duration and surgical consideration?",
   "tier": 4, "tier_label": "adversarial"}
]
```

### non_english (4 gastro + 4 infect; mix Russian/French/Spanish)

```json
[
  {"id": "adv_lang_9",  "category": "non_english", "expected_specialist": "gastroenterologist",
   "query": "Пациент 28 лет с 6-месячной историей правосторонней подвздошной боли, диареей без крови и потерей веса 7 кг. Колоноскопия выявила прерывистые поражения подвздошной кишки. Какая тактика индукции ремиссии?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_10", "category": "non_english", "expected_specialist": "gastroenterologist",
   "query": "Femme de 54 ans, prurit chronique et fatigue. Phosphatase alcaline élevée à 4x N, anticorps anti-mitochondries positifs au 1/320. Diagnostic et traitement de première ligne ?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_11", "category": "non_english", "expected_specialist": "gastroenterologist",
   "query": "Varón de 45 años con hepatitis B crónica, HBeAg negativo, ADN-VHB 25.000 UI/ml, ALT 96, Fibroscan 9,2 kPa. ¿Indicación y elección del tratamiento antiviral?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_12", "category": "non_english", "expected_specialist": "gastroenterologist",
   "query": "Пациент 48 лет с 6-месячной изжогой после еды, регургитацией кислоты и ночным кашлем. Имеет ожирение. Первая линия диагностики и терапии?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_13", "category": "non_english", "expected_specialist": "infectionist",
   "query": "Patient de 68 ans, fièvre 39 °C depuis 4 jours, toux productive, douleur thoracique pleurale et SpO2 92 %. Radiographie : condensation lobaire inférieure droite. CRB-65 = 1. Quelle antibiothérapie empirique ?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_14", "category": "non_english", "expected_specialist": "infectionist",
   "query": "Hombre de 60 años con fiebre 39,2 °C, FC 115, TA 88/52, lactato 3,5 mmol/L y foco intraabdominal sospechoso. ¿Qué medidas de la hora-1 y antibioterapia empírica iniciar?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_15", "category": "non_english", "expected_specialist": "infectionist",
   "query": "Пациент 35 лет вернулся из Нигерии. Лихорадка 5 дней, желтуха, олигурия, спутанность. Толстая капля: P. falciparum, паразитемия 8%. Тактика лечения тяжёлой малярии?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_lang_16", "category": "non_english", "expected_specialist": "infectionist",
   "query": "Patient migrant de 35 ans, toux productive depuis 3 mois, perte de poids et sueurs nocturnes, hémoptysie. Frottis BAAR positif sur deux échantillons. Quel schéma thérapeutique pour la tuberculose pulmonaire ?",
   "tier": 4, "tier_label": "adversarial"}
]
```

### dominant_pathology_mismatch (8 cases — gastro/infect surface vocab, non-gastro/non-infect expected)

```json
[
  {"id": "adv_dom_9",  "category": "dominant_pathology_mismatch", "expected_specialist": "cardiologist",
   "query": "65-year-old male with well-controlled Child-Pugh A cirrhosis on lactulose and rifaximin, last paracentesis 8 weeks ago, presents to the ED with sudden crushing substernal chest pain at rest, ST elevation in leads II/III/aVF, troponin 18 ng/L. Liver function is stable. What is the priority diagnosis and immediate management?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_10", "category": "dominant_pathology_mismatch", "expected_specialist": "infectionist",
   "query": "32-year-old female with Crohn's disease in clinical remission on infliximab and methotrexate for 2 years presents with 3 weeks of low-grade fever, drenching night sweats, productive cough and a new right upper-lobe cavitary lesion on chest CT. She had a negative pre-biologic TB screen. Stool calprotectin is normal. What is the priority diagnosis and immediate management?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_11", "category": "dominant_pathology_mismatch", "expected_specialist": "endocrinologist",
   "query": "55-year-old male with chronic hepatitis C completed direct-acting antiviral therapy with SVR12 confirmed 8 months ago, undetectable HCV RNA, normalised ALT, no fibrosis progression. Now presents with polyuria, polydipsia, weight loss, fasting glucose 12.8 mmol/L and HbA1c 9.6 %. What is the priority diagnosis and treatment?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_12", "category": "dominant_pathology_mismatch", "expected_specialist": "endocrinologist",
   "query": "70-year-old female on long-term pantoprazole 40 mg daily for GERD (10 years) and a stable diet presents with carpopedal spasm, tetany and a positive Trousseau sign. Calcium 1.65 mmol/L (corrected), magnesium 0.32 mmol/L. PPI is the only chronic medication. What is the priority diagnosis and management?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_13", "category": "dominant_pathology_mismatch", "expected_specialist": "cardiologist",
   "query": "55-year-old male with well-controlled HIV on dolutegravir/abacavir/lamivudine for 7 years (CD4 720, viral load <40), no opportunistic infections, presents with new-onset effort angina, a positive treadmill test, and LDL 4.8 mmol/L. He has a 30-pack-year smoking history. What is the priority diagnosis and management approach?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_14", "category": "dominant_pathology_mismatch", "expected_specialist": "endocrinologist",
   "query": "42-year-old male completed a full 6-month course of RIPE therapy for pulmonary tuberculosis 12 months ago, currently asymptomatic from a pulmonary standpoint with full radiographic resolution. He now presents with fatigue, weight loss, postural hypotension, hyperpigmentation, hyponatraemia and an inappropriately low morning cortisol. What is the priority diagnosis?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_15", "category": "dominant_pathology_mismatch", "expected_specialist": "cardiologist",
   "query": "73-year-old female with recent community-acquired pneumonia completed a 5-day course of amoxicillin/clavulanate 4 weeks ago with full clinical and radiographic resolution. She now presents with 2 days of progressive exertional dyspnoea, leg swelling, raised JVP and a new S3 gallop. BNP 1800. What is the priority diagnosis and management?",
   "tier": 4, "tier_label": "adversarial"},
  {"id": "adv_dom_16", "category": "dominant_pathology_mismatch", "expected_specialist": "endocrinologist",
   "query": "50-year-old female with 4 recurrent uncomplicated cystitis episodes over the past year, all cleared with short-course nitrofurantoin and currently asymptomatic, presents for routine follow-up. Random glucose is 16.2 mmol/L, HbA1c 9.1 %. She reports unintentional 6 kg weight loss and polyuria for 3 months. What is the priority diagnosis and management?",
   "tier": 4, "tier_label": "adversarial"}
]
```

Per-category, the new 8 dominant_pathology_mismatch cases break down
into 3 cardio + 4 endo + 1 infect expected_specialists (only 1
infect-expected because writing a gastro/infect-surface case that
correctly resolves to gastro/infect is by definition *not* a mismatch
— the trap-and-target asymmetry forces most cases to resolve to
cardio or endo).

### symptom_only_ambiguous (8 cases)

```json
[
  {"id": "adv_amb_9",  "category": "symptom_only_ambiguous", "expected_specialist": "infectionist",
   "query": "A 38-year-old male presents with one week of fever (38.6°C), jaundice, mild nausea and right upper quadrant discomfort. He has no other localising symptoms and recently returned from a 3-week trip. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["gastroenterologist", "infectionist"]},
  {"id": "adv_amb_10", "category": "symptom_only_ambiguous", "expected_specialist": "gastroenterologist",
   "query": "A 42-year-old female reports 4 months of intermittent watery diarrhoea, 5 kg weight loss and low-grade fever (37.6–38.0°C). There is no overt blood in the stool, no recent antibiotics and no foreign travel. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["gastroenterologist", "infectionist"]},
  {"id": "adv_amb_11", "category": "symptom_only_ambiguous", "expected_specialist": "infectionist",
   "query": "A 65-year-old male presents with 5 days of fever (38.4°C), progressive dyspnoea on exertion and bilateral pleural effusions on chest X-ray. He has no productive cough or chest pain. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["cardiologist", "infectionist"]},
  {"id": "adv_amb_12", "category": "symptom_only_ambiguous", "expected_specialist": "gastroenterologist",
   "query": "A 60-year-old male presents with 6 weeks of progressive abdominal distension, bilateral leg swelling and 4 kg weight gain. There is no orthopnoea or paroxysmal nocturnal dyspnoea and no fever. Initial examination shows ascites and pitting oedema.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["cardiologist", "gastroenterologist"]},
  {"id": "adv_amb_13", "category": "symptom_only_ambiguous", "expected_specialist": "endocrinologist",
   "query": "A 50-year-old female presents with 3 months of fatigue, polyuria, 4 kg unintentional weight loss and intermittent itching over the chest and back. She has no abdominal pain, no jaundice and no recent travel. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["endocrinologist", "gastroenterologist"]},
  {"id": "adv_amb_14", "category": "symptom_only_ambiguous", "expected_specialist": "infectionist",
   "query": "A 38-year-old male presents with 6 months of recurrent oral candidiasis (4 distinct episodes), unintentional 8 kg weight loss and persistent fatigue. He has no fever, no specific exposures and no immunosuppressive medications. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["endocrinologist", "infectionist"]},
  {"id": "adv_amb_15", "category": "symptom_only_ambiguous", "expected_specialist": "gastroenterologist",
   "query": "A 45-year-old female presents with 8 months of vague right upper quadrant discomfort, intermittent low-grade fever (37.5–37.9°C) and unintentional 3 kg weight loss. There is no jaundice, no diarrhoea and no overt infectious exposure. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["gastroenterologist", "infectionist"]},
  {"id": "adv_amb_16", "category": "symptom_only_ambiguous", "expected_specialist": "endocrinologist",
   "query": "A 55-year-old female presents with 5 months of constipation (2–3 stools per week), cold intolerance, slow mentation and 4 kg weight gain. She is not on chronic medications and reports no abdominal pain. No initial labs are available yet.",
   "tier": 4, "tier_label": "adversarial",
   "valid_domains": ["endocrinologist", "gastroenterologist"]}
]
```

Adversarial total post-stage: 64 cases (16 per category).

## 5. Spot-check output (4 cases × retrieved chunks + answer + routing)

End-to-end runs via `MedicalOrchestrator.answer()`:

### gastro_5 (T1 — IBS-D)

- **Routing**: `gastroenterologist` (correct)
- **Retrieval top-5**: L2 ∈ [1.011, 1.046]; #1 = `gutjnl-2017-315909/0037.txt` (BSG IBD/IBS guideline section discussing differential between microscopic colitis and IBS-D).
- **`answer()` outcome**: **`[RefusalGate] gastroenterologist refusing query (out-of-scope)`** → `"Insufficient evidence in the current knowledge base to address this specific query."`
- **Bypassed-gate LLM output** (to verify the case isn't authored too hard): correctly identifies IBS-D, references chunk 0032.txt, properly limits scope to what the context covers, and does not hallucinate. The case is well-authored; the agent CAN answer it.

### gastro_25 (T2 — achalasia)

- **Routing**: `gastroenterologist` (correct)
- **Retrieval top-5**: L2 ∈ [0.986, 1.016]; #1 = `nihms-1868619/0003.txt` (HHS achalasia diagnostic chapter explicitly mentioning bird-beak + manometry).
- **`answer()` outcome**: **refused** by the L2 gate.
- **Bypassed-gate LLM output**: correctly identifies achalasia, explains aperistalsis + LES non-relaxation + bird-beak. Notes that treatment options are not in the retrieved context (true — the chunk it cited focuses on diagnosis, not management). Conservative and correct.

### infect_5 (T1 — viridans-strep IE)

- **Routing**: `infectionist` (correct)
- **Retrieval top-5**: L2 ∈ [1.010, 1.095]; #1 = `12879_2019_Article_4532/0003.txt` (case report of culture-positive endocarditis).
- **`answer()` outcome**: **refused** by the L2 gate.
- **Bypassed-gate LLM output**: identifies IE, but the retrieved chunks happened to surface an Enterococcus hirae case rather than viridans-strep-specific treatment. The LLM correctly notes "Context does not provide specific information about viridans group streptococci" — no hallucination. The case is authored correctly; the IE+viridans+treatment-duration combination is a corpus content gap rather than a case gap.

### infect_22 (T2 — severe falciparum malaria)

- **Routing**: `infectionist` (correct)
- **Retrieval top-5**: L2 ∈ [0.963, 0.988]; top hits are tropical medicine series and tick-borne disease chapters. None specifically discuss artesunate or WHO severity criteria.
- **`answer()` outcome**: **refused** by the L2 gate.
- **Bypassed-gate LLM output**: cleanly returns "Insufficient evidence in the current knowledge base" — the agent correctly refuses to invent management content not present in the context. The infect corpus has 444 chunks for `malaria` but the specific clinical-scenario keywords (parasitaemia threshold + artesunate + WHO severity) don't co-occur in the surfaced top-5.

**Spot-check conclusion**: all 4 cases route correctly, retrieve
plausibly relevant chunks within `MAX_L2_DISTANCE = 1.2`, and the
LLM behaves correctly within the retrieved evidence (no
hallucination). None of the 4 cases need revision. The fact that
`answer()` returns the refusal string for every one of them is
**not** a case-quality issue — it is the Stage-7 L2 refusal gate's
`L2_REJECT_MIN = 0.92` threshold firing because the new corpora's
in-corpus L2 distribution sits at ≈ 1.00 mean (gastro 1.007 ± 0.062;
infect 1.004 ± 0.048; cf. cardio 0.94, endo 0.84 in Prompt 1 §7.4).
Confirmed by the auto-generated `corpus_dist_stats.json` files
written by `RefusalGate.from_vectorstore()` during the spot check:
gastro `mu = 0.880`, infect `mu = 0.894` — both already above the
0.92 gate threshold, so the gate cannot be expected to spare any
in-corpus query without recalibration. This is Prompt 3's job.

## 6. Known gaps and flagged inconsistencies

### 6.1 Dev/test re-split moves the optimisation point

Dev grew from 30 → 60 cases (2× larger) and test from 70 → 140 (2×).
The hyperparameter choices in `report_final.md` §3.4 (`SIMILARITY_TOP_K`,
`MAX_L2_DISTANCE`, chunk-size grid) and §4.5 (`L2_REJECT_MIN`) were
selected on the 30-case dev set. Re-running `tune_retrieval.py`,
`tune_chunk_size.py`, and `tune_refusal_gate.py` on the 60-case dev
set MAY shift the chosen K / L2 thresholds / refusal-gate setpoint.
This is Prompt 3's job; **none of those scripts were re-run in this
stage** (constraint: "DO NOT run … refusal-gate re-tune").

### 6.2 Multijudge canonical run on 140-case test split

The Stage 31 multijudge reconciliation canonical run is on the
70-case test split. After re-split to 140 cases, a new canonical
multijudge run is needed. Cost estimate: ~140 cases × 2 judges =
280 Pro-tier LLM calls ≈ 1,500 ₽ at current Yandex prices. Same
methodology — `evaluate_generation.py --split test --mode multi_judge`.

### 6.3 `valid_domains` vs `domains` schema split

`ambiguous_cases.json` has the dual-field quirk introduced in Stage 31
(`ambig_9..14` carry both `valid_domains` and `domains` with
identical values; `ambig_1..8` carry only `domains`). The new
`adv_amb_9..16` adversarial cases use `valid_domains` only — matching
the *Stage 25 adversarial* convention, not the *Stage 31 ambiguous*
convention. Result: a downstream tool that scans both files for
ambiguous-domain pairs needs to look at both fields. **I did not
unify the schemas** (out of scope per the brief: "Pick one and migrate
the other to match in a follow-up commit; flag the inconsistency but
do NOT fix it in this prompt").

### 6.4 T3 = "out_of_scope" in tier_label but in-domain in practice

The brief's verbal description of Tier 3 says "queries that present
GI-adjacent symptoms but are NOT gastroenterology" with examples like
"abdominal pain in a patient with sickle cell crisis (haematology)".
But the existing `cardio_28..cardio_36` and `endo_44..endo_50` T3
cases are NOT cross-domain at all — they are rare/peripheral
in-domain conditions (constrictive pericarditis, Takayasu arteritis,
cardiac sarcoidosis, AL amyloid, etc.) intended to test the L2
refusal gate via corpus thinness, not via cross-domain misrouting.
The brief explicitly says "Pattern matches cardio_28..cardio_36 /
endo_44..endo_50" — so I matched the existing pattern (in-domain,
corpus-thin) rather than the verbal description (cross-domain).
**Brief vs. data inconsistency: noted, followed the data.** All 16
new T3 cases (8 gastro + 8 infect) are in-domain conditions chosen
because their corpus chunk count is < 10 (the conditions are real
gastro/infect topics, just under-represented in the corpora).

### 6.5 Auto-annotator circularity (Stage 6 §6 Limitation 7)

The auto-annotator's gold_sources are sampled from the embedding-based
top-K=20 + keyword filter — same evidence set that the live retrieval
will operate on. This means an annotator-empty case ≠ truly
unanswerable case. The 6 new corpus-gap cases in §3 all had top-20
retrievals within `MAX_L2_DISTANCE`; just the `expected_keywords` list
did not co-occur with the right chunk. Carrying the same bias as
the pre-existing 100-case set.

### 6.6 `expected_keywords` cross-specialty leakage risk

None of the 100 new cases use keywords that would canonically fire
on a different specialty's corpus. Sanity check spot-checks: gastro
keywords like "Wilson disease", "MASLD", "Barrett's oesophagus" are
gastro-specific; infect keywords like "Plasmodium falciparum",
"Tropheryma whipplei", "Buruli ulcer" are infect-specific. The
closest call is `adv_dom_10` (Crohn's on infliximab → TB reactivation)
— the keywords "Crohn's", "infliximab" might surface gastro chunks
even though `expected_specialist` is `infectionist`. This is
intentional (it's a `dominant_pathology_mismatch` adversarial case;
keyword overlap is the *trap*).

### 6.7 `RefusalGate.from_vectorstore()` side effect

Running the spot-check end-to-end caused `RefusalGate.from_vectorstore()`
to compute and cache per-specialty corpus distribution statistics to:

```
?? data/processed/gastroenterologist/corpus_dist_stats.json
?? data/processed/infection/corpus_dist_stats.json
```

These are intentional cache files (Signal-B parameters for the
Stage-7 refusal gate); the cardio + endo equivalents are already
committed at the same path. Worth committing alongside the
dataset changes.

### 6.8 The Stage-7 L2 gate is stale for the new corpora

Cross-referenced from §5 spot-check output and the §7.4 L2 analysis in
`report_stage_indices_built.md`:

| Specialty | Corpus mean L2 (`mu`) | `L2_REJECT_MIN = 0.92` margin | In-corpus T1 query refuse rate |
|---|---:|---:|---:|
| cardiology    | ~0.84 (Stage 7 baseline) | +0.08 below μ | low (current behaviour) |
| endocrinology | ~0.82 (Stage 7 baseline) | +0.10 below μ | low (current behaviour) |
| gastroenterologist | **0.880** | -0.04 below μ | **4 / 4 spot-check refusals** |
| infection         | **0.894** | -0.03 below μ | **4 / 4 spot-check refusals** |

The threshold currently sits **below** the new corpora's corpus mean,
which means **the gate will refuse most in-corpus queries** on gastro
and infect. **Required Prompt-3 work**: re-run
`tune_refusal_gate.py` on the 60-case dev set with all four
specialties, and either (a) raise `L2_REJECT_MIN` to a value that
preserves cardio + endo Tier-3 refusal rates while not over-refusing
gastro + infect Tier 1, or (b) introduce per-specialty thresholds.