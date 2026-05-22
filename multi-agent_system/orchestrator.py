import json
import re

import openai

from agents import SpecialistAgent
from agents.registry import AGENT_REGISTRY
from settings import ROUTING_MODEL, YANDEX_PROJECT_ID, client


EMERGENCY_PATTERNS = re.compile(
    r"\b("
    r"heart attack|cardiac arrest|stroke|can'?t breathe|cannot breathe|"
    r"losing consciousness|lost consciousness|call 911|call an ambulance|"
    r"severe bleeding|choking|seizure|anaphylaxis|overdose|"
    r"chest pain right now|dying|i'?m going to die"
    r")\b",
    re.IGNORECASE,
)

TREATMENT_PATTERNS = re.compile(
    r"\b("
    r"prescribe me|prescribe .+ for me|what medication should i take|"
    r"give me a dosage|write me a prescription|what drug should i use|"
    r"recommend a dose|how much .+ should i take"
    r")\b",
    re.IGNORECASE,
)

EMERGENCY_MESSAGE = (
    "Your query describes a potential medical emergency.\n"
    "This AI system is NOT a substitute for emergency medical care.\n\n"
    "Please call your local emergency number (e.g. 911, 112, 103) "
    "or go to the nearest emergency room IMMEDIATELY."
)

TREATMENT_MESSAGE = (
    "This system cannot prescribe medications or recommend specific dosages.\n"
    "For treatment decisions, please consult a licensed healthcare professional "
    "who can review your full medical history."
)


class MedicalOrchestrator:
    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = knowledge_base_dir
        # `knowledge_base_dir` is kept for backward compatibility with the prior
        # constructor signature; the per-specialty paths come from the registry.
        self.agents = {key: SpecialistAgent(**cfg) for key, cfg in AGENT_REGISTRY.items()}
        # Strict allow-list for routing — `route()` accepts the LLM's output
        # only if it canonicalises to one of these. No alias coercion.
        self.allowed_specialists = list(self.agents.keys())

    def safety_check(self, question: str) -> str | None:
        if EMERGENCY_PATTERNS.search(question):
            return EMERGENCY_MESSAGE
        if TREATMENT_PATTERNS.search(question):
            return TREATMENT_MESSAGE
        return None

    def _routing_system_prompt(self) -> str:
        scopes = []
        for key in self.allowed_specialists:
            agent = self.agents[key]
            scopes.append(f"  - {key!r}: {agent.domain_scope}")
        scope_block = "\n".join(scopes)
        allowed = ", ".join(repr(s) for s in self.allowed_specialists)
        return (
            "You are a medical orchestrator. Determine which specialist should "
            "handle the request. The available specialists and their domain "
            "scopes are:\n"
            f"{scope_block}\n\n"
            "Output a single JSON object with key `specialist` whose value is "
            f"one of: {allowed}. Do not output any other text. "
            'Example: {"specialist": "cardiologist"}.'
        )

    def _parse_router_output(self, raw: str) -> str | None:
        """Parse the router's response and validate against the allow-list.

        Returns the canonical specialty string if valid, `None` otherwise.
        No silent aliasing — `"cardiology"`, `"surgeon"`, or any other
        non-allowed string returns `None`.
        """
        if raw is None:
            return None
        text = raw.strip()
        # First try strict JSON parsing (the default path: prompt asks for JSON
        # and Yandex's response_format=json_object enforces it server-side).
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                spec = str(obj.get("specialist", "")).strip().lower()
                if spec in self.allowed_specialists:
                    return spec
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def route(self, question):
        """LLM-based routing.

        Uses Yandex's `response_format={"type": "json_object"}` (verified
        supported on `gpt://{folder}/yandexgpt/latest` during Stage 19 probe).
        Falls back to plain `chat.completions.create` if Yandex ever rejects
        the parameter — in either case the response is parsed via
        `_parse_router_output` and validated against `self.allowed_specialists`.

        Returns:
          - the canonical specialty string on success;
          - `"__error__:validation"` if the LLM produced output that did not
            parse into a valid `{"specialist": ...}` JSON object whose value
            is in the allow-list — NOT silently re-mapped;
          - `"__error__:authentication"` / `"__error__:rate_limit"` /
            `"__error__:connection"` on the respective `openai` exceptions;
          - `"__error__:unknown:..."` on any other exception.
        """
        common_kwargs = {
            "model": ROUTING_MODEL,
            "messages": [
                {"role": "system", "content": self._routing_system_prompt()},
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "extra_headers": {"x-folder-id": YANDEX_PROJECT_ID},
        }
        try:
            try:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    **common_kwargs,
                )
            except (openai.BadRequestError, TypeError):
                # Yandex / OpenAI client doesn't accept response_format on this
                # endpoint — fall back. The JSON prompt is still in place, so
                # the model's output will still usually be a JSON object;
                # `_parse_router_output` validates it either way.
                response = client.chat.completions.create(**common_kwargs)
            raw = response.choices[0].message.content
            specialist = self._parse_router_output(raw)
            if specialist is None:
                # Do NOT silently re-map. Surface the failure.
                return "__error__:validation"
            return specialist
        except openai.AuthenticationError:
            return "__error__:authentication"
        except openai.RateLimitError:
            return "__error__:rate_limit"
        except openai.APIConnectionError:
            return "__error__:connection"
        except Exception as e:
            return f"__error__:unknown:{e}"

    def answer(self, question):
        if not question or not question.strip():
            return "System", "Please enter a valid medical question.", "No evidence retrieved."
        if len(question.strip()) < 3:
            return "System", "Your query is too short. Please provide more detail.", "No evidence retrieved."

        safety_result = self.safety_check(question)
        if safety_result:
            return "Safety Gateway", safety_result, "No evidence retrieved."

        specialist = self.route(question)

        if specialist.startswith("__error__:"):
            error_type = specialist.split(":")[1]
            if error_type == "authentication":
                return "Error", "API authentication failed. Please check your YANDEX_API_KEY.", "No evidence retrieved."
            elif error_type == "rate_limit":
                return "Error", "API rate limit reached. Please wait a moment and try again.", "No evidence retrieved."
            elif error_type == "connection":
                return "Error", "Unable to connect to the Yandex Foundation Models API.", "No evidence retrieved."
            elif error_type == "validation":
                return "Error", "Routing failed: the LLM did not return a recognised specialist.", "No evidence retrieved."
            else:
                return "Error", "An unexpected error occurred while routing your query.", "No evidence retrieved."

        print("specialist: ", specialist)
        agent = self.agents.get(specialist)
        if agent is None:
            return "Error", "Could not determine the specialist.", "No evidence retrieved."
        return agent.answer(question)
