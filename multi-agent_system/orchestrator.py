import re

import openai
from pydantic import BaseModel

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


class RouteDecision(BaseModel):
    specialist: str


class MedicalOrchestrator:
    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = knowledge_base_dir
        # `knowledge_base_dir` is kept for backward compatibility with the prior
        # constructor signature; the per-specialty paths come from the registry.
        self.agents = {key: SpecialistAgent(**cfg) for key, cfg in AGENT_REGISTRY.items()}

    # ----- backward-compat aliases -----
    # Many evaluation / annotation scripts written before Stage 8 read
    # `orchestrator.cardiologist` / `orchestrator.endocrinologist` directly.
    # These properties forward to `self.agents[...]` so those callers keep
    # working unchanged; new code should use `self.agents[key]`.
    @property
    def cardiologist(self):
        return self.agents.get("cardiologist")

    @property
    def endocrinologist(self):
        return self.agents.get("endocrinologist")

    # ----- safety + routing -----

    def safety_check(self, question: str) -> str | None:
        if EMERGENCY_PATTERNS.search(question):
            return EMERGENCY_MESSAGE
        if TREATMENT_PATTERNS.search(question):
            return TREATMENT_MESSAGE
        return None

    def _routing_system_prompt(self) -> str:
        """Build the router's system prompt from the registry.

        Important: the prompt text must remain textually identical to the
        pre-refactor wording so that `evaluate_routing.py --split dev` produces
        the same routing decisions. With the current 2-specialist registry,
        joining `self.agents.keys()` with " or " reproduces the original
        "cardiologist or endocrinologist" verbatim. The `domain_scope` field
        is exported by the registry for future use (e.g. when a third
        specialist is added and the routing instructions need to expand);
        it is intentionally NOT inlined here so the LLM's routing behaviour
        on the current 2-specialist set stays byte-identical to Stage 7.
        """
        specialist_list = " or ".join(self.agents.keys())
        return (
            "You are a medical orchestrator. "
            "Determine which specialist should handle the request: "
            f"{specialist_list}. "
            "Respond strictly in one word."
        )

    def route(self, question):
        try:
            response = client.chat.completions.create(
                model=ROUTING_MODEL,
                messages=[
                    {"role": "system", "content": self._routing_system_prompt()},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=10,
                extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
            )
            specialist = response.choices[0].message.content.strip().lower()
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
                return "Error", "API authentication failed. Please check your OPENAI_API_KEY.", "No evidence retrieved."
            elif error_type == "rate_limit":
                return "Error", "API rate limit reached. Please wait a moment and try again.", "No evidence retrieved."
            elif error_type == "connection":
                return "Error", "Unable to connect to the OpenAI API.", "No evidence retrieved."
            else:
                return "Error", "An unexpected error occurred while routing your query.", "No evidence retrieved."

        print("specialist: ", specialist)
        agent = self.agents.get(specialist)
        if agent is None:
            return "Error", "Could not determine the specialist.", "No evidence retrieved."
        return agent.answer(question)
