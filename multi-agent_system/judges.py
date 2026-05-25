
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import openai
import requests

from settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SECONDARY_JUDGE_API_KEY,
    YANDEX_API_KEY,
    YANDEX_PROJECT_ID,
    client as yandex_client,
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert Clinical QA Evaluator. Your task is to determine "
    "if a Generated Answer introduces medical facts that are NOT supported "
    "by the Retrieved Context.\n\n"
    "JUDGMENT RULES:\n"
    "1. Mark as FAITHFUL if the answer:\n"
    "   - Paraphrases or summarizes information from the Context.\n"
    "   - Uses clinical synonyms for terms in the Context "
    "(e.g., 'heart attack' for 'myocardial infarction').\n"
    "   - Draws a direct logical inference clearly supported by the Context.\n"
    "   - States 'Insufficient evidence' or declines to answer.\n"
    "2. Mark as HALLUCINATION only if the answer introduces:\n"
    "   - Specific drug names, dosages, or treatment protocols NOT in the Context.\n"
    "   - Specific lab values, thresholds, or diagnostic criteria NOT in the Context.\n"
    "   - Statistics, percentages, or epidemiological facts NOT in the Context.\n"
    "   - A specific diagnosis NOT mentioned or clearly implied by the Context.\n\n"
    "Respond ONLY with the single word 'FAITHFUL' or 'HALLUCINATION'."
)


@dataclass
class JudgeConfig:
    name: str
    provider: str
    model_id: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    request_interval_s: float = 0.0

    @classmethod
    def from_uri(cls, name: str, uri: str) -> "JudgeConfig":
        if uri.startswith("yandex:"):
            return cls(name=name, provider="yandex", model_id=uri[len("yandex:"):])
        if uri.startswith("openrouter:"):
            if not OPENROUTER_API_KEY:
                raise EnvironmentError(
                    f"{name} judge configured for OpenRouter but OPENROUTER_API_KEY is unset."
                )
            return cls(
                name=name,
                provider="openrouter",
                model_id=uri[len("openrouter:"):],
                api_key=OPENROUTER_API_KEY,
                endpoint=OPENROUTER_BASE_URL,
                request_interval_s=2.0,
            )
        if uri.startswith("http:"):
            if not SECONDARY_JUDGE_API_KEY:
                raise EnvironmentError(
                    f"{name} judge configured for http: but SECONDARY_JUDGE_API_KEY is unset."
                )
            return cls(
                name=name,
                provider="http",
                model_id="(see endpoint)",
                api_key=SECONDARY_JUDGE_API_KEY,
                endpoint=uri[len("http:"):],
                request_interval_s=2.0,
            )
        raise ValueError(
            f"Unrecognised judge URI prefix in {name}={uri!r}. "
            "Expected one of: yandex:, openrouter:, http:"
        )


MAX_RETRIES = 5


class JudgeStats:

    def __init__(self) -> None:
        self.http_errors: int = 0
        self.exhausted: int = 0
        self.successes: int = 0
        self.auth_errors: int = 0
        self.rate_limit_errors: int = 0
        self.connection_errors: int = 0
        self.timeout_errors: int = 0
        self.other_errors: int = 0


def _user_prompt(query: str, context: str, generated_answer: str) -> str:
    return (
        f"--- CLINICAL QUERY ---\n{query}\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context}\n\n"
        f"--- GENERATED ANSWER ---\n{generated_answer}\n\n"
        "Is the Generated Answer faithful to the Retrieved Context, or does it "
        "hallucinate specific medical facts not present in the Context?"
    )


def _parse_judgement(raw: str) -> Optional[bool]:
    upper = raw.strip().upper()
    if "FAITHFUL" in upper and "HALLUCINATION" not in upper:
        return True
    if "HALLUCINATION" in upper:
        return False
    return None


def _judge_yandex(cfg: JudgeConfig, query: str, context: str, answer: str,
                  case_id: str, stats: JudgeStats) -> Optional[bool]:
    for attempt in range(MAX_RETRIES):
        try:
            response = yandex_client.chat.completions.create(
                model=cfg.model_id,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(query, context, answer)},
                ],
                temperature=0.0,
                max_tokens=64,
                extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
            )
        except openai.AuthenticationError as e:
            stats.auth_errors += 1
            print(f"  [{cfg.name}] auth error on {case_id}: {e}")
            return None
        except openai.RateLimitError as e:
            stats.rate_limit_errors += 1
            print(f"  [{cfg.name}] rate limit on {case_id} (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(min(2 ** attempt, 30))
            continue
        except openai.APIConnectionError as e:
            stats.connection_errors += 1
            print(f"  [{cfg.name}] connection error on {case_id} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(min(2 ** attempt, 30))
            continue
        except openai.APITimeoutError as e:
            stats.timeout_errors += 1
            print(f"  [{cfg.name}] timeout on {case_id} (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(min(2 ** attempt, 30))
            continue
        except Exception as e:
            stats.other_errors += 1
            print(f"  [{cfg.name}] UNEXPECTED {type(e).__name__} on {case_id}: {e}")
            raise

        try:
            content = response.choices[0].message.content or ""
            verdict = _parse_judgement(content)
        except (IndexError, AttributeError, TypeError) as e:
            stats.other_errors += 1
            print(f"  [{cfg.name}] malformed response on {case_id}: {type(e).__name__}: {e}")
            return None

        if verdict is not None:
            stats.successes += 1
            return verdict
        print(f"  [{cfg.name}] unparseable verdict on {case_id}: {content!r}")
        return None

    stats.exhausted += 1
    print(f"  [{cfg.name}] EXHAUSTED retries on {case_id} — returning None")
    return None


def _judge_openai_compatible(cfg: JudgeConfig, query: str, context: str, answer: str,
                             case_id: str, stats: JudgeStats) -> Optional[bool]:
    assert cfg.endpoint and cfg.api_key
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if cfg.provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/Lunciare/multi-agent-medical-rag"
        headers["X-Title"] = "multi-agent-medical-rag faithfulness judge"
    payload = {
        "model": cfg.model_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(query, context, answer)},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    for attempt in range(MAX_RETRIES):
        if cfg.request_interval_s:
            time.sleep(cfg.request_interval_s)
        try:
            res = requests.post(cfg.endpoint, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                body = res.json()
                content = body["choices"][0]["message"]["content"] or ""
                verdict = _parse_judgement(content)
                if verdict is not None:
                    stats.successes += 1
                    return verdict
                print(f"  [{cfg.name}] unparseable response for {case_id}: {content!r}")
                return None
            if res.status_code == 429 or res.status_code >= 500:
                stats.http_errors += 1
                print(f"  [{cfg.name}] HTTP {res.status_code} on {case_id} "
                      f"(attempt {attempt+1}/{MAX_RETRIES}): {res.text[:200]}")
                time.sleep(min(2 ** attempt, 30))
            else:
                stats.http_errors += 1
                print(f"  [{cfg.name}] HTTP {res.status_code} on {case_id}: "
                      f"{res.text[:300]} — returning None")
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError) as e:
            stats.http_errors += 1
            print(f"  [{cfg.name}] network error on {case_id} "
                  f"(attempt {attempt+1}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            time.sleep(min(2 ** attempt, 30))
    stats.exhausted += 1
    print(f"  [{cfg.name}] EXHAUSTED retries on {case_id} — returning None")
    return None


def judge_faithfulness(query: str, context: str, generated_answer: str,
                       judge_config: JudgeConfig, case_id: str,
                       stats: JudgeStats) -> Optional[bool]:
    if judge_config.provider == "yandex":
        return _judge_yandex(judge_config, query, context, generated_answer, case_id, stats)
    if judge_config.provider in ("openrouter", "http"):
        return _judge_openai_compatible(judge_config, query, context, generated_answer,
                                        case_id, stats)
    raise ValueError(f"Unknown judge provider: {judge_config.provider!r}")
