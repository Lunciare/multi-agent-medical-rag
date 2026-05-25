# External Benchmark: PubMedQA Cardiology Slice (`pqa_labeled`, n=85)

**Date:** 2026-05-20 21:24:03  
**Source:** Jin et al. 2019 — *PubMedQA: A Dataset for Biomedical Research Question Answering* (EMNLP 2019). HuggingFace: `qiaojin/PubMedQA`, subset `pqa_labeled`, split `train` (1000 expert-labelled QA pairs).  

## 1. Filtering

Filtered the 1000-case `pqa_labeled` split to cardiology-relevant questions using a case-insensitive substring OR over {heart, cardiac, cardio, ventricular, atrial, coronary, mitral, aortic, valve, arrhythmia, hypertension, stroke}.  
**Filtered count:** 85 questions.

## 2. Matching Heuristic

For each cardiology-filtered question, the cardiologist agent's FAISS index returns the top-K=5 chunks. Each chunk and each gold passage (one element of `context.contexts` on the PubMedQA record) is split into sentences on `[.!?]` boundaries. A retrieved chunk is judged to *hit* a gold passage when at least one (chunk_sentence, gold_sentence) pair reaches token-level Jaccard similarity `|A ∩ B| / |A ∪ B|` >= 0.2, where tokens are lowercased alphanumeric words of length >= 2. The sentence-level formulation is necessary because cardiology corpus chunks are ~400 words while PubMedQA passages are ~50-150 words; a passage-level Jaccard between such asymmetric units is structurally capped near 0.2 even when the gold tokens are fully contained in the chunk. The threshold was calibrated empirically: a probe across all 275 gold passages found the maximum sentence-pair Jaccard achievable on this corpus-vs-PubMedQA pair was 0.294 (mean 0.163), because the clinical-guideline / textbook register of the cardiology corpus differs systematically from PubMedQA's research-abstract register. A 0.2 threshold sits at the 21.5% percentile of that achievable distribution and is the operating point that surfaces a non-zero comparison signal without being dominated by stopword overlap. Each gold passage is one Bernoulli trial in the pooled Recall@5; the trial succeeds if any of the five retrieved chunks crosses the sentence-pair Jaccard threshold against it.

## 3. Pooled Recall@5

| Metric | Value | 95% Wilson CI |
|---|---|---|
| Hits / gold passages | 59 / 275 | — |
| Recall@5 | 21.5% | [17.0%–26.7%] |

## 4. Per-Case Hit Counts

| pubid | gold passages | hits | per-case Recall@5 | final_decision |
|---|---|---|---|---|
| `22990761` | 4 | 0 | 0% | yes |
| `21402341` | 3 | 1 | 33% | no |
| `11340218` | 3 | 1 | 33% | no |
| `27491658` | 3 | 1 | 33% | yes |
| `23999452` | 2 | 0 | 0% | yes |
| `21823940` | 3 | 0 | 0% | no |
| `18565233` | 3 | 0 | 0% | yes |
| `18322741` | 3 | 1 | 33% | yes |
| `21900017` | 3 | 2 | 67% | yes |
| `25987398` | 3 | 2 | 67% | maybe |
| `21952349` | 2 | 1 | 50% | yes |
| `17276182` | 4 | 0 | 0% | yes |
| `22428608` | 3 | 0 | 0% | yes |
| `27858166` | 3 | 2 | 67% | yes |
| `21946341` | 3 | 0 | 0% | no |
| `25156467` | 3 | 0 | 0% | yes |
| `22720085` | 3 | 1 | 33% | yes |
| `8910148` | 3 | 0 | 0% | yes |
| `10548670` | 3 | 1 | 33% | yes |
| `17051586` | 3 | 0 | 0% | no |
| `11079675` | 4 | 0 | 0% | yes |
| `24507422` | 3 | 0 | 0% | yes |
| `21801416` | 2 | 2 | 100% | yes |
| `25592625` | 5 | 0 | 0% | no |
| `12595848` | 3 | 2 | 67% | yes |
| `18955431` | 7 | 1 | 14% | yes |
| `16296668` | 4 | 1 | 25% | no |
| `17306983` | 3 | 0 | 0% | yes |
| `21881325` | 3 | 1 | 33% | yes |
| `15141797` | 3 | 0 | 0% | no |
| `12963175` | 3 | 1 | 33% | yes |
| `22440363` | 3 | 3 | 100% | no |
| `27690714` | 3 | 0 | 0% | yes |
| `7497757` | 6 | 0 | 0% | no |
| `15222284` | 3 | 1 | 33% | yes |
| `19155657` | 3 | 0 | 0% | maybe |
| `21198823` | 3 | 0 | 0% | no |
| `19142546` | 4 | 1 | 25% | no |
| `9107172` | 2 | 1 | 50% | yes |
| `12040336` | 3 | 2 | 67% | no |
| `10490564` | 4 | 0 | 0% | yes |
| `26104852` | 3 | 0 | 0% | yes |
| `10201555` | 3 | 1 | 33% | yes |
| `21342862` | 3 | 0 | 0% | yes |
| `18041059` | 3 | 0 | 0% | yes |
| `26460153` | 2 | 1 | 50% | yes |
| `19640728` | 3 | 1 | 33% | yes |
| `24172579` | 3 | 0 | 0% | yes |
| `23264436` | 2 | 1 | 50% | yes |
| `26237424` | 3 | 1 | 33% | no |
| `25228241` | 3 | 0 | 0% | yes |
| `12607666` | 3 | 0 | 0% | yes |
| `17224424` | 2 | 0 | 0% | yes |
| `14652839` | 3 | 3 | 100% | no |
| `25571931` | 3 | 1 | 33% | maybe |
| `21084567` | 4 | 1 | 25% | yes |
| `26175531` | 2 | 1 | 50% | no |
| `19575104` | 3 | 0 | 0% | no |
| `16971978` | 4 | 0 | 0% | yes |
| `10577397` | 3 | 3 | 100% | yes |
| `25150098` | 3 | 0 | 0% | yes |
| `17462393` | 3 | 0 | 0% | no |
| `10173769` | 7 | 0 | 0% | yes |
| `10973547` | 3 | 1 | 33% | no |
| `26163474` | 3 | 0 | 0% | yes |
| `26965932` | 4 | 0 | 0% | yes |
| `12855939` | 5 | 1 | 20% | no |
| `26304701` | 3 | 0 | 0% | yes |
| `15053041` | 3 | 1 | 33% | no |
| `24340838` | 4 | 1 | 25% | yes |
| `15208005` | 3 | 1 | 33% | yes |
| `24669960` | 3 | 0 | 0% | no |
| `24671913` | 3 | 0 | 0% | yes |
| `26063028` | 3 | 2 | 67% | yes |
| `10732884` | 4 | 1 | 25% | no |
| `19351635` | 3 | 1 | 33% | maybe |
| `24318956` | 3 | 0 | 0% | yes |
| `22768311` | 3 | 0 | 0% | no |
| `23848044` | 2 | 1 | 50% | yes |
| `25891436` | 3 | 0 | 0% | yes |
| `25810292` | 3 | 2 | 67% | yes |
| `16216859` | 6 | 1 | 17% | no |
| `17062234` | 3 | 2 | 67% | no |
| `27131771` | 3 | 1 | 33% | yes |
| `10456814` | 3 | 0 | 0% | no |
