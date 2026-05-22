# Stage 3 Report: Tiered Evaluation Infrastructure

## 1. What Was Changed
- `multi-agent_system/tests/data/golden_dataset_100.json`: Renamed to `golden_dataset.json` to act as the new primary evaluation dataset.
- `multi-agent_system/tests/evaluate_retrieval.py`: Added a smoke test (`--smoke-test`) to validate dataset integrity, implemented `(domain, tier)` counters, and added tables for Tier Hit Rate and Tier 3 Fallback Behaviour.
- `multi-agent_system/tests/evaluate_routing.py`: Added `(domain, tier)` accuracy tracking, terminal table printing, and updated the markdown report generation to include the tier breakdown.
- `multi-agent_system/tests/evaluate_chunk_relevance.py`: Implemented `(domain, tier)` counters and tier summary table reporting.
- `multi-agent_system/tests/evaluate_generation.py`: Added tier tracking, safely handled the new `agent.answer()` tuple return type, and implemented specific tracking for Tier 3 'Insufficient evidence' fallbacks.
- `README.md`: Updated the Evaluation Results dataset description and tabular structures to reflect the 3-tier 100-case dataset, and added the `--smoke-test` command.

## 2. Smoke Test Output
```text
Running Smoke Test...
Smoke Test Passed! Dataset is valid and correctly formatted.
```

## 3. Evaluation Results — Verbatim Terminal Output

### Retrieval Evaluation Summary
```text
============================================================
  Retrieval Evaluation Results
============================================================
  Domain                 Hits  Total   Hit Rate
  -------------------- ------ ------ ----------
  cardiologist             43     50     86.0%
  endocrinologist          47     50     94.0%
  -------------------- ------ ------ ----------
  OVERALL                  90    100     90.0%
============================================================

============================================================
  Retrieval Hit Rate — By Tier
============================================================
  Domain               Tier   Label           Hits  Total    Hit Rate
  -------------------- ------ ------------- ------ ------  ----------
  cardiologist         1      core              27     27     100.0%
  cardiologist         2      peripheral        11     14      78.6%
  cardiologist         3      out_of_scope       5      9      55.6%
  endocrinologist      1      core              26     27      96.3%
  endocrinologist      2      peripheral        14     16      87.5%
  endocrinologist      3      out_of_scope       7      7     100.0%
============================================================

============================================================
  Tier 3 (Out-of-Scope) — Fallback Behaviour
============================================================
  cardio_10       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_28       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_29       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_30       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_31       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_32       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_46       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_47       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  cardio_48       Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_39         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_40         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_41         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_42         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_43         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_44         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
  endo_50         Chunks retrieved: 5  ⚠️ ADJACENT CONTENT
============================================================
```

### Routing Evaluation Summary
```text
============================================================
  Routing Evaluation — Golden Dataset
============================================================
  Domain                Correct    Total   Accuracy
  -------------------- -------- -------- ----------
  cardiologist               50       50    100.0%
  endocrinologist            49       50     98.0%
  -------------------- -------- -------- ----------
  OVERALL                    99      100     99.0%
============================================================

============================================================
  Routing Accuracy — By Tier
============================================================
  Domain               Tier   Label         Correct   Total    Accuracy
  -------------------- ------ ------------- ------- -------  ----------
  cardiologist         1      core               27      27     100.0%
  cardiologist         2      peripheral         14      14     100.0%
  cardiologist         3      out_of_scope        9       9     100.0%
  endocrinologist      1      core               27      27     100.0%
  endocrinologist      2      peripheral         15      16      93.8%
  endocrinologist      3      out_of_scope        7       7     100.0%
============================================================
```

### Generation Evaluation Summary
```text
============================================================
  Generation Evaluation Results (Faithfulness)
============================================================
  Domain               Faithful  Total      Score
  -------------------- -------- ------ ----------
  cardiologist               50     50    100.0%
  endocrinologist            50     50    100.0%
  -------------------- -------- ------ ----------
  OVERALL                   100    100    100.0%
============================================================

============================================================
  Faithfulness — By Tier
============================================================
  Domain               Tier   Label         Faithful  Total  Faithfulness
  -------------------- ------ ------------- -------- ------  ------------
  cardiologist         1      core                27     27       100.0%
  cardiologist         2      peripheral          14     14       100.0%
  cardiologist         3      out_of_scope         9      9       100.0%
  endocrinologist      1      core                27     27       100.0%
  endocrinologist      2      peripheral          16     16       100.0%
  endocrinologist      3      out_of_scope         7      7       100.0%
============================================================

============================================================
  Tier 3 Fallback Responses
============================================================
  0 / 16 cases returned 'Insufficient evidence' message (expected behaviour).
============================================================
```
*(Note: Per-query generation logs were truncated in this report for brevity, but the final metrics are preserved identically as outputted).*

## 4. Results Interpretation
a) **Tier 1 Performance**: Tier 1 Hit Rate held perfectly in Cardiology (100%) and very high in Endocrinology (96.3%), validating that the core knowledge base remains highly functional for standard cases. Faithfulness remained at a perfect 100%.
b) **Tier 2 Hit Rate**: Tier 2 Hit Rate saw an expected drop compared to Tier 1, scoring 78.6% for Cardiology and 87.5% for Endocrinology. This gap highlights the difficulty of retrieving thinly-covered conditions (peripheral cases) and provides a clear signal for targeted KB expansion.
c) **Tier 3 Retrieval Behaviour**: Interestingly, NO Tier 3 cases returned zero chunks. All 16 cases returned 5 chunks, flagging them as "⚠️ ADJACENT CONTENT". Because the system retrieves 5 chunks, it means the `MAX_L2_DISTANCE` threshold wasn't strict enough to reject semantic neighbors for out-of-scope queries. As a result, 0/16 cases triggered the "Insufficient evidence" fallback, since the LLM was fed and faithfully generated an answer based on this adjacent content.
d) **Tier 3 Routing**: There were no unexpected routing failures on Tier 3 cases. The routing model successfully determined the expected specialist (100% accuracy) despite the queries being out-of-scope for the specific knowledge bases.

## 5. Open Questions
- **Retrieval L2 Distance Threshold**: Since all 16 Tier 3 queries managed to pull 5 chunks under the `MAX_L2_DISTANCE` threshold, do we need to tighten the L2 cutoff? Currently, the safety fallback is never triggering because adjacent content is always being passed to the generator.
- **Safety Fallback Prompting**: Because the system generates an answer based on adjacent (but technically "unhelpful" for the exact query) content, it passes the faithfulness evaluation (it didn't hallucinate outside the context), but it fails the clinical intent of "I don't know". How should we adjust the prompt to reject clinically irrelevant adjacent content?

## 6. Commit Message
`[eval] feature: implement 3-tier analysis and update golden dataset to 100 cases`
