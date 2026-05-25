# Error & Failure Analysis

This document provides a detailed breakdown of the 4 specific failures encountered across the evaluation of the multi-agent medical RAG system. By analysing the exact failure modes (routing, retrieval, and generation), we can identify targeted architectural and content-level fixes.

---

## 1. Routing Failure: `cardio_15`
**Metric affected:** Routing Accuracy

* **The Query:** "A trauma patient presents with muffled heart sounds, jugular venous distention, and hypotension (Beck's triad). An urgent bedside ultrasound shows fluid in the pericardial space. What acute intervention is required?" (Expected: cardiologist / cardiac tamponade)
* **What the system did:** The LLM router dispatched the query to the `surgeon` agent.
* **Why it failed:** The prompt included "surgeon" as a theoretical stub agent. The presence of the word "trauma" and the implicit need for an acute procedural intervention (pericardiocentesis/thoracotomy) caused the LLM to confidently, but incorrectly within the bounds of our implemented RAG pipelines, route it to surgery.
* **The Fix:** Removed unimplemented stub agents (`surgeon`, `dermatologist`) from the router's system prompt. The prompt now strictly constrains the output space to `cardiologist` or `endocrinologist`.

## 2. Routing Failure: `endo_12`
**Metric affected:** Routing Accuracy

* **The Query:** "A 45-year-old male presents with therapy-resistant hypertension and hypokalemia. Plasma renin activity is suppressed while serum aldosterone is significantly elevated. What is the diagnosis?" (Expected: endocrinologist / primary aldosteronism)
* **What the system did:** The LLM router dispatched the query to the `cardiologist` agent.
* **Why it failed:** The query foregrounds "therapy-resistant hypertension." Hypertension is overwhelmingly statistically associated with cardiology. The LLM router failed to weight the specific lab values (suppressed renin, elevated aldosterone) as indicative of a secondary endocrine cause (Conn's syndrome), opting instead for a superficial symptom match.
* **The Fix:** To fix this, the router's system prompt needs few-shot examples demonstrating how to handle secondary presentations of common symptoms. For example, explicitly showing that "hypertension + hypokalemia + aldosterone abnormalities → endocrinologist."

## 3. Retrieval Miss: `cardio_10`
**Metric affected:** Retrieval Hit Rate

* **The Query:** "A 55-year-old male presents to the ED with severe, tearing chest pain radiating to his back. His blood pressure is 190/110 mmHg. A widened mediastinum is seen on chest X-ray. What is the immediate next step in management?" (Expected keywords: aortic dissection, ct angiography)
* **What the system did:** The query successfully routed to the cardiologist agent. However, the FAISS vector database retrieved generic chunks discussing chest pain and hypertension, none of which contained the expected diagnostic keywords.
* **Why it failed:** This is a **content gap**. The cardiology knowledge base (raw text corpus) simply lacks a dedicated chapter or detailed section on acute aortic dissection. The embedding model and FAISS search worked correctly by finding the closest semantic matches (chest pain guidelines), but the specific required knowledge did not exist in the database.
* **The Fix:** Expand the raw `data/raw/cardiology/` corpus to include texts covering acute aortic syndromes and vascular emergencies, then re-run the `build_cardio_index.py` pipeline.

## 4. Generation Failure: Historical Faithfulness Miss
**Metric affected:** Faithfulness / Generation Quality

* **The Context:** Prior to the final system rebuild, the generation pipeline scored 29/30 (96.7%) on faithfulness, with one cardiology case flagged as a `HALLUCINATION` by the LLM-as-a-judge.
* **What the system did:** The cardiologist agent generated an answer that introduced a specific diagnostic protocol. Because this protocol was medically accurate but *not present in the retrieved context chunks*, the strict LLM judge correctly flagged it as unfaithful to the evidence.
* **Why it failed:** The root cause was poor retrieval, not a poorly prompted generator. The old cardiology FAISS index included `KEYWORDS:` metadata injected directly into the text, which polluted the embeddings and caused the retrieval of suboptimal chunks. Deprived of the exact evidence needed, the generator's parametric memory overrode the strict prompt instructions.
* **The Fix:** The cardiology index was rebuilt using keyword-stripping (extracting `KEYWORDS:` into metadata rather than embedding them). This improved retrieval quality, providing the generator with the correct evidence. On the rebuilt indices, faithfulness achieved 30/30 (100%), confirming that high-quality retrieval prevents hallucinations.
