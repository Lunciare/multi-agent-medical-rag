from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from langchain_core.documents import Document


class BaseMedicalAgent(ABC):

    refusal_gate: Optional[object] = None

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        ...

    @abstractmethod
    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        ...

    @abstractmethod
    def answer(self, question: str) -> Tuple[str, str, str]:
        ...

    def refuse(self, query: str) -> bool:
        if self.refusal_gate is None:
            return False
        return bool(self.refusal_gate.refuse(query))
