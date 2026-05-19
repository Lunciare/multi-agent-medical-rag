import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

YANDEX_API_KEY: str | None = os.getenv("YANDEX_API_KEY")
YANDEX_PROJECT_ID: str | None = os.getenv("YANDEX_PROJECT_ID")
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"

if not YANDEX_API_KEY:
    raise EnvironmentError(
        "YANDEX_API_KEY environment variable is not set. "
        "Please add it to your .env file or export it:\n"
        "  export YANDEX_API_KEY='...'"
    )

if not YANDEX_PROJECT_ID:
    raise EnvironmentError(
        "YANDEX_PROJECT_ID environment variable is not set. "
        "Please add it to your .env file or export it:\n"
        "  export YANDEX_PROJECT_ID='...'"
    )

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url=YANDEX_BASE_URL,
)

ROUTING_MODEL = f"gpt://{YANDEX_PROJECT_ID}/yandexgpt/latest"
AGENT_MODEL = f"gpt://{YANDEX_PROJECT_ID}/yandexgpt/latest"
EMBEDDING_MODEL = "text-search-doc/latest"
EMBEDDING_QUERY_MODEL = "text-search-query/latest"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "processed", "cardiology")
ENDO_KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "processed", "endocrinology")

SIMILARITY_TOP_K = 5
MAX_L2_DISTANCE = 1.2
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 30

PRIMARY_JUDGE_PROVIDER: str = f"yandex:gpt://{YANDEX_PROJECT_ID}/yandexgpt/latest"

SECONDARY_JUDGE_PROVIDER: str | None = os.getenv("SECONDARY_JUDGE_PROVIDER") or None
TERTIARY_JUDGE_PROVIDER: str | None = os.getenv("TERTIARY_JUDGE_PROVIDER") or None

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
SECONDARY_JUDGE_API_KEY: str | None = os.getenv("SECONDARY_JUDGE_API_KEY")
