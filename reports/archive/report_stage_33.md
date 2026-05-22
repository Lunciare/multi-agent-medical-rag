# Stage 33 — Per-Exception Handling in `_judge_yandex`

(Filename note: next sequential number — Stages 23 through 32 are
already taken: dict-access migration, `domain_scope` routing prompt,
adversarial routing, README refresh, MRR bootstrap CI, registry
schema-validation tests, real Gradio UI test, §4.5 architectural
framing, multijudge reconciliation, dead-code / inactive-config
cleanup.)

## 1. What Was Changed

`_judge_yandex` in [`multi-agent_system/judges.py`](../multi-agent_system/judges.py)
previously wrapped its API call in a single `except Exception` that
incremented a generic `http_errors` counter for every failure type —
authentication errors, rate limits, connection drops, timeouts,
malformed responses, and genuine bugs all looked the same in the
markdown summary. This stage splits the catches per `openai` exception
class, adds per-category counters to `JudgeStats`, and renders the new
counters in the `evaluate_generation.py` markdown summary so an eval
run that hits a transient error surfaces *which* kind of error it was
without needing the stdout log.

Three categories of change:

1. **`JudgeStats`** extended with five new counters (`auth_errors`,
   `rate_limit_errors`, `connection_errors`, `timeout_errors`,
   `other_errors`). The legacy `http_errors` and `exhausted` counters
   are kept for backward compatibility with `_judge_openai_compatible`
   (the OpenRouter / generic HTTP path that still uses them).
2. **`_judge_yandex`** body replaced with spec's per-exception
   structure: separate `except` clauses for
   `openai.AuthenticationError`, `RateLimitError`, `APIConnectionError`,
   `APITimeoutError`, and a bare `except Exception` that increments
   `other_errors` and re-`raise`s so unknown bugs surface in CI rather
   than being silently swallowed. Response-parsing exceptions
   (`IndexError`, `AttributeError`, `TypeError`) are caught in a
   separate inner `try` and counted as `other_errors`. The retry-sleep
   logic (`min(2 ** attempt, 30)`) is preserved for the three
   transient-error categories; auth errors are non-retryable per the
   spec's intent.
3. **`evaluate_generation.py`** Configured Judges markdown table
   header rewritten from
   `| Role | Provider | Model URI | HTTP errors | Retries exhausted | Successful calls |`
   to
   `| Role | Provider | Model URI | Auth | Rate-limit | Conn | Timeout | Other | Successes |`
   to render the new per-category counters. Row format updated
   correspondingly.

## 2. Diff for `_judge_yandex` (and surrounding edits to `judges.py`)

```diff
diff --git a/multi-agent_system/judges.py b/multi-agent_system/judges.py
index 705b08c3f..c4c090657 100644
--- a/multi-agent_system/judges.py
+++ b/multi-agent_system/judges.py
@@ -20,6 +20,7 @@ import time
 from dataclasses import dataclass
 from typing import Optional

+import openai
 import requests

 from settings import (
@@ -106,6 +107,11 @@ class JudgeStats:
         self.http_errors: int = 0
         self.exhausted: int = 0
         self.successes: int = 0
+        self.auth_errors: int = 0
+        self.rate_limit_errors: int = 0
+        self.connection_errors: int = 0
+        self.timeout_errors: int = 0
+        self.other_errors: int = 0


 def _user_prompt(query: str, context: str, generated_answer: str) -> str:
@@ -141,18 +147,46 @@ def _judge_yandex(cfg: JudgeConfig, query: str, context: str, answer: str,
                 max_tokens=64,
                 extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
             )
-            verdict = _parse_judgement(response.choices[0].message.content or "")
-            if verdict is not None:
-                stats.successes += 1
-                return verdict
-            print(f"  [{cfg.name}] unparseable response for {case_id}: "
-                  f"{response.choices[0].message.content!r}")
-            return None
-        except Exception as e:
-            stats.http_errors += 1
-            print(f"  [{cfg.name}] error on {case_id} (attempt {attempt+1}/{MAX_RETRIES}): "
-                  f"{type(e).__name__}: {e}")
+        except openai.AuthenticationError as e:
+            stats.auth_errors += 1
+            print(f"  [{cfg.name}] auth error on {case_id}: {e}")
+            return None  # auth errors are not retryable
+        except openai.RateLimitError as e:
+            stats.rate_limit_errors += 1
+            print(f"  [{cfg.name}] rate limit on {case_id} (attempt {attempt+1}/{MAX_RETRIES})")
             time.sleep(min(2 ** attempt, 30))
+            continue
+        except openai.APIConnectionError as e:
+            stats.connection_errors += 1
+            print(f"  [{cfg.name}] connection error on {case_id} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
+            time.sleep(min(2 ** attempt, 30))
+            continue
+        except openai.APITimeoutError as e:
+            stats.timeout_errors += 1
+            print(f"  [{cfg.name}] timeout on {case_id} (attempt {attempt+1}/{MAX_RETRIES})")
+            time.sleep(min(2 ** attempt, 30))
+            continue
+        except Exception as e:
+            # Unexpected exception — log and re-raise so the bug surfaces in CI.
+            stats.other_errors += 1
+            print(f"  [{cfg.name}] UNEXPECTED {type(e).__name__} on {case_id}: {e}")
+            raise
+
+        # Parse the response, separately:
+        try:
+            content = response.choices[0].message.content or ""
+            verdict = _parse_judgement(content)
+        except (IndexError, AttributeError, TypeError) as e:
+            stats.other_errors += 1
+            print(f"  [{cfg.name}] malformed response on {case_id}: {type(e).__name__}: {e}")
+            return None
+
+        if verdict is not None:
+            stats.successes += 1
+            return verdict
+        print(f"  [{cfg.name}] unparseable verdict on {case_id}: {content!r}")
+        return None
+
     stats.exhausted += 1
     print(f"  [{cfg.name}] EXHAUSTED retries on {case_id} — returning None")
     return None
```

The replacement is character-for-character per spec, with the only
extension being the trailing `else`-less fall-through (the `continue`
on retryable exceptions skips the parse block on this iteration; a
successful API call falls through to the parse block; auth + other
errors short-circuit with `return None` / `raise` before the parse
block). The spec's body composes cleanly with the existing
`for attempt in range(MAX_RETRIES):` loop, with the existing
`stats.exhausted += 1` / final `return None` outside the loop, and
with the unchanged `JUDGE_SYSTEM_PROMPT` / `_user_prompt` / `_parse_judgement`
helpers.

## 3. Smoke-Test Output

Command (verbatim per spec):

```python
python -c "
from judges import JudgeStats
s = JudgeStats()
assert s.auth_errors == 0 and s.rate_limit_errors == 0
assert s.connection_errors == 0 and s.timeout_errors == 0
assert s.other_errors == 0
print('JudgeStats extension OK')
"
```

Stdout:

```
JudgeStats extension OK
```

All five new counters initialise to `0`. Module import succeeds (the
new `import openai` resolves cleanly — `openai` was already a
transitive dep of `client` from `settings.py`).

## 4. New Markdown Summary Header (live run, verbatim)

Command:

```
cd multi-agent_system
SECONDARY_JUDGE_PROVIDER="yandex:gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest" \
  python tests/evaluate_generation.py --split test --mode multi_judge
```

Result file: `reports/faithfulness_multijudge_2026-05-21.md` (351.4 s,
116 judge calls). The Configured Judges section now reads:

```markdown
## Configured Judges

| Role | Provider | Model URI | Auth | Rate-limit | Conn | Timeout | Other | Successes |
|---|---|---|---|---|---|---|---|---|
| yandex_primary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt/latest` | 0 | 0 | 0 | 0 | 0 | 58 |
| secondary | yandex | `gpt://b1ga5vl107uu7uqguvp3/yandexgpt-lite/latest` | 0 | 0 | 0 | 0 | 0 | 58 |
```

The header expanded from 6 columns → 9 columns; the row payload
expanded from `{http_errors} | {exhausted} | {successes}` → `{auth}
| {rate_limit} | {conn} | {timeout} | {other} | {successes}` as the
spec requires.

## 5. Per-Counter Statement for the Live Run

**All five new counters are zero on both judges in this run.**

| Counter | yandex_primary | secondary |
|---|---|---|
| `auth_errors` | 0 | 0 |
| `rate_limit_errors` | 0 | 0 |
| `connection_errors` | 0 | 0 |
| `timeout_errors` | 0 | 0 |
| `other_errors` | 0 | 0 |
| `successes` | 58 | 58 |

The run was clean: 116 judge calls (58 cases × 2 judges, 12 Tier 3
cases excluded as `fallback=True`), no retries triggered, no
authentication / rate-limit / connection / timeout / parse-error
exceptions. Both judges returned `successes == 58 == Total Judged`,
matching the per-judge faithfulness rates (`yandex_primary` 100.0%,
`secondary` 98.3%) that the §4.4 / Stage 31 reconciliation document
already cites as canonical.

**Interpretation when this matters next:** the next eval run that
*does* hit a transient error will surface the category in the table
(e.g. a Yandex API timeout will increment the `Timeout` column rather
than the legacy `HTTP errors` umbrella column). On the OpenRouter /
generic-HTTP judge path (`_judge_openai_compatible`, unchanged), the
legacy `http_errors` and `exhausted` counters continue to be
incremented and are kept on `JudgeStats` for backward compatibility,
but they are no longer rendered in the markdown table — a future stage
can decide whether to (a) keep them as JSON sidecar fields, (b) add
them back as additional table columns, or (c) refactor
`_judge_openai_compatible` to use the same per-category counters.

## 6. Headline Numbers (Sanity Check — Should Be Unchanged)

The refactor is a logging / accounting change only, not a semantics
change. Comparing the 2026-05-21 11:35:17 run to the canonical
2026-05-19 run:

| Metric | 2026-05-19 (canonical) | 2026-05-21 (post-Stage-33) | Match? |
|---|---|---|---|
| `yandex_primary` FAITHFUL | 58 / 58 = 100.0% | 58 / 58 = 100.0% | ✅ |
| `secondary` FAITHFUL | 57 / 58 = 98.3% | 57 / 58 = 98.3% | ✅ |
| Minimum-judge FAITHFUL | 57 / 58 = 98.3% | 57 / 58 = 98.3% | ✅ |
| Pair n / agreements / κ | 58 / 57 / 0.000 | 58 / 57 / 0.000 | ✅ |
| Single disagreement case | `cardio_40` (FAITHFUL vs HALLUCINATION) | `cardio_40` (FAITHFUL vs HALLUCINATION) | ✅ |
| Tier 3 fallback rows | 12 / 15 | 12 / 15 | ✅ |

Every headline number matches. The new file
`faithfulness_multijudge_2026-05-21.md` (and the matching raw CSV)
will not displace the 2026-05-19 canonical run cited in §4.4 — it's
written to disk under its dated filename for traceability and as a
witness that the refactored `_judge_yandex` produces identical
verdicts on identical prompts.

## 7. Files Touched

- `multi-agent_system/judges.py` — `import openai`, `JudgeStats` +5
  counters, `_judge_yandex` body rewritten per spec
- `multi-agent_system/tests/evaluate_generation.py` — Configured
  Judges header + row format updated to 9 columns
- `reports/faithfulness_multijudge_2026-05-21.md` — generated by the
  live re-run (new table format visible in §2 of that file)
- `reports/faithfulness_multijudge_raw_2026-05-21.csv` — generated by
  the live re-run (byte-identical to the 2026-05-19 / 2026-05-20 raw
  CSVs — the underlying verdicts are stable; see
  `reports/multijudge_reconciliation.md`)
- `reports/report_stage_33.md` — this stage report (new)
