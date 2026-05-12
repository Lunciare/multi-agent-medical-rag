import concurrent.futures
import time
from typing import Dict, List

import requests
from langchain_core.embeddings import Embeddings

from settings import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_MODEL,
    YANDEX_API_KEY,
    YANDEX_PROJECT_ID,
)

MAX_WORKERS: int = 3
REQUEST_DELAY: float = 0.25
MAX_RETRIES: int = 7
EMBEDDING_URL: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"


class YandexNativeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.doc_uri: str = f"emb://{YANDEX_PROJECT_ID}/{EMBEDDING_MODEL}"
        self.query_uri: str = f"emb://{YANDEX_PROJECT_ID}/{EMBEDDING_QUERY_MODEL}"
        self.url: str = EMBEDDING_URL
        self.headers: Dict[str, str] = {
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": str(YANDEX_PROJECT_ID),
        }

    def _embed(self, text: str, model_uri: str) -> List[float]:
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                res = requests.post(
                    self.url,
                    headers=self.headers,
                    json={"modelUri": model_uri, "text": text},
                    timeout=60,
                )
                if res.status_code == 200:
                    return res.json().get("embedding", [])
                elif res.status_code == 429:
                    time.sleep(min(2 ** attempt, 30))
                else:
                    raise ValueError(f"API error {res.status_code}: {res.text}")
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OSError,
            ):
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = [[] for _ in range(len(texts))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fmap = {
                ex.submit(self._embed, t, self.doc_uri): i
                for i, t in enumerate(texts)
            }
            for f in concurrent.futures.as_completed(fmap):
                results[fmap[f]] = f.result()
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, self.query_uri)
