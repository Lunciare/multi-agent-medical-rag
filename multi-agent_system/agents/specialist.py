
from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from agents.base import BaseMedicalAgent
from embeddings import YandexNativeEmbeddings
from refusal_gate import RefusalGate
from settings import (
    AGENT_MODEL,
    L2_REJECT_MIN,
    MAX_L2_DISTANCE,
    REFUSAL_GATE_SIGNAL,
    SIMILARITY_TOP_K,
    YANDEX_PROJECT_ID,
    client,
)

logger = logging.getLogger(__name__)


REFUSAL_RESPONSE = (
    "Insufficient evidence in the current knowledge base to address this specific query."
)
NO_EVIDENCE = "No evidence retrieved."


class SpecialistAgent(BaseMedicalAgent):

    def __init__(self, *, name: str, folder_path: str, role_prompt: str,
                 domain_scope: str) -> None:
        self.name = name
        self.folder_path = folder_path
        self.role_prompt = role_prompt
        self.domain_scope = domain_scope
        self.embeddings = YandexNativeEmbeddings()
        self._refusal_gate = None

        faiss_save_path = os.path.join(folder_path, "faiss_index")
        if os.path.exists(faiss_save_path):
            logger.info("Loading cached FAISS index from %s", faiss_save_path)
            self.vectorstore = FAISS.load_local(
                faiss_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS vector store loaded successfully for %s", self.name)
            return

        logger.info("No cached FAISS index found at %s. Reading documents...", faiss_save_path)
        documents = self._load_documents(folder_path)
        if not documents:
            raise ValueError(
                f"No documents found in {folder_path}. "
                "Ensure the directory contains .json or .txt chunk files."
            )

        logger.info("Building FAISS index for %d document chunks (%s)...",
                    len(documents), self.name)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info("FAISS vector store built successfully for %s", self.name)

        logger.info("Saving FAISS index to %s for faster future startups...",
                    faiss_save_path)
        self.vectorstore.save_local(faiss_save_path)


    @staticmethod
    def _load_documents(folder_path: str) -> List[Document]:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Knowledge-base directory not found: {folder_path}"
            )

        documents: List[Document] = []
        for root, _dirs, files in os.walk(folder_path):
            if "faiss_index" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)

                if file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            continue
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    metadata = data.get("metadata", {})
                    documents.append(
                        Document(page_content=text, metadata=metadata)
                    )

                elif file.endswith(".txt") and file.lower() != "summary.txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        continue
                    lines = text.split("\n")
                    clean_lines = []
                    keywords = ""
                    for line in lines:
                        if line.startswith("KEYWORDS:"):
                            keywords = line.replace("KEYWORDS:", "").strip()
                        else:
                            clean_lines.append(line)
                    clean_text = "\n".join(clean_lines).strip()
                    if not clean_text:
                        continue
                    parts = file_path.replace("\\", "/").split("/")
                    category = parts[-3] if len(parts) >= 3 else "unknown"
                    documents.append(
                        Document(
                            page_content=clean_text,
                            metadata={
                                "source_file": file,
                                "category": category,
                                "keywords": keywords,
                            },
                        )
                    )
        return documents


    @property
    def refusal_gate(self):
        if self._refusal_gate is None:
            specialty = os.path.basename(self.folder_path.rstrip("/"))
            self._refusal_gate = RefusalGate.from_vectorstore(
                self.vectorstore,
                specialty=specialty,
                processed_dir=self.folder_path,
                l2_reject_min=L2_REJECT_MIN,
                corpus_dist_k=1.0,
                signal=REFUSAL_GATE_SIGNAL,
                top_k=SIMILARITY_TOP_K,
            )
        return self._refusal_gate


    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=SIMILARITY_TOP_K
        )
        return [(doc, score) for doc, score in docs_and_scores
                if score <= MAX_L2_DISTANCE]

    def answer(self, question: str) -> Tuple[str, str, str]:
        if self.refuse(question):
            logger.info("RefusalGate: %s refusing query (out-of-scope)", self.name.lower())
            return self.name, REFUSAL_RESPONSE, NO_EVIDENCE

        valid_docs_and_scores = self.retrieve(question)

        logger.info("Retrieved %d chunks for %s", len(valid_docs_and_scores), self.name.lower())
        context_parts = []
        for i, (doc, l2_score) in enumerate(valid_docs_and_scores, 1):
            source = doc.metadata.get("source_file", "unknown")
            category = doc.metadata.get("category", "unknown")
            sim_score = max(0.0, 1.0 - (l2_score / MAX_L2_DISTANCE))
            preview = doc.page_content[:150].replace("\n", " ")
            logger.debug("  [%d] source=%s | category=%s | confidence=%.1f%% | %s...",
                         i, source, category, sim_score * 100, preview)
            context_parts.append(
                f"[Source: {source}, Confidence: {sim_score:.1%}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)
        prompt_user = f"Evidence Context:\n{context}\n\nClinical Query:\n{question}"

        try:
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": self.role_prompt},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.0,
                max_tokens=1024,
                extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
            )
            final_answer = response.choices[0].message.content.strip()

            if valid_docs_and_scores:
                evidence_md = ""
                for i, (doc, l2_score) in enumerate(valid_docs_and_scores, 1):
                    source = doc.metadata.get("source_file", "unknown")
                    category = doc.metadata.get("category", "unknown")
                    sim_score = max(0.0, 1.0 - (l2_score / MAX_L2_DISTANCE))
                    evidence_md += (
                        f"### Chunk {i} | Source: {source} | Category: {category} | "
                        f"Confidence: {sim_score:.1%}\n"
                        f"```text\n{doc.page_content}\n```\n\n"
                    )
            else:
                evidence_md = NO_EVIDENCE

            return self.name, final_answer, evidence_md
        except Exception as e:
            return (
                self.name,
                (
                    f"The {self.name.lower()} agent encountered an error while "
                    f"generating a response. Please try again later. "
                    f"(Detail: {type(e).__name__}: {e})"
                ),
                "Error retrieving evidence.",
            )
