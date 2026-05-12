import re
import openai
from pydantic import BaseModel
from agents.cardiologist import CardiologistAgent
from agents.endocrinologist import EndocrinologistAgent
from settings import client, ROUTING_MODEL, YANDEX_PROJECT_ID, ENDO_KNOWLEDGE_BASE_DIR


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
        self.cardiologist = CardiologistAgent(folder_path=knowledge_base_dir)
        self.endocrinologist = EndocrinologistAgent(folder_path=ENDO_KNOWLEDGE_BASE_DIR)

    def safety_check(self, question: str) -> str | None:
        if EMERGENCY_PATTERNS.search(question):
            return EMERGENCY_MESSAGE
        if TREATMENT_PATTERNS.search(question):
            return TREATMENT_MESSAGE
        return None

    def route(self, question):
        try:
            response = client.chat.completions.create(
                  model=ROUTING_MODEL,
                  messages=[
                      {"role": "system", "content": (
                          "You are a medical orchestrator. "
                          "Determine which specialist should handle the request: "
                          "cardiologist or endocrinologist. "
                          "Respond strictly in one word."
                      )},
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
        if specialist == "cardiologist":
            resp, ev = self.cardiologist.answer(question)
            return "Cardiologist", resp, ev
        elif specialist == "endocrinologist":
            resp, ev = self.endocrinologist.answer(question)
            return "Endocrinologist", resp, ev
        else:
            return "Error", "Could not determine the specialist.", "No evidence retrieved."
