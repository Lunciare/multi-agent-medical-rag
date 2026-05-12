import os
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from agents.base import BaseMedicalAgent
from embeddings import YandexNativeEmbeddings
from settings import (
    client,
    AGENT_MODEL,
    SIMILARITY_TOP_K,
    YANDEX_PROJECT_ID,
    MAX_L2_DISTANCE,
)


class CardiologistAgent(BaseMedicalAgent):

    def __init__(self, folder_path: str):
        self.embeddings = YandexNativeEmbeddings()

        faiss_save_path = os.path.join(folder_path, "faiss_index")

        if os.path.exists(faiss_save_path):
            print(f"Loading cached FAISS index from {faiss_save_path}")
            self.vectorstore = FAISS.load_local(
                faiss_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("FAISS vector store loaded successfully")
            return

        print("No cached FAISS index found. Reading documents...")
        documents = []

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Knowledge-base directory not found: {folder_path}"
            )

        for root, dirs, files in os.walk(folder_path):
            if "faiss_index" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)

                if file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            text = data.get("text", "").strip()
                            metadata = data.get("metadata", {})
                            if text:
                                documents.append(
                                    Document(page_content=text, metadata=metadata)
                                )
                        except json.JSONDecodeError:
                            continue

                elif file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
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

        if not documents:
            raise ValueError(
                f"No documents found in {folder_path}. "
                "Ensure the directory contains .json or .txt chunk files."
            )


        print(f"Building FAISS index for {len(documents)} document chunks...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("FAISS vector store built successfully")

        print(f"Saving FAISS index to {faiss_save_path} for faster future startups...")
        self.vectorstore.save_local(faiss_save_path)

    def answer(self, question):
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=SIMILARITY_TOP_K)

        valid_docs_and_scores = [(doc, score) for doc, score in docs_and_scores if score <= MAX_L2_DISTANCE]

        print(f"\n{'='*60}")
        print(f"Retrieved {len(valid_docs_and_scores)} chunks:")
        print(f"{'='*60}")
        context_parts = []
        for i, (doc, l2_score) in enumerate(valid_docs_and_scores, 1):
            source = doc.metadata.get("source_file", "unknown")
            category = doc.metadata.get("category", "unknown")
            sim_score = max(0.0, 1.0 - (l2_score / MAX_L2_DISTANCE))
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"\n  [{i}] source={source}  |  category={category}  |  confidence={sim_score:.1%}")
            print(f"      {preview}...")
            context_parts.append(f"[Source: {source}, Confidence: {sim_score:.1%}]\n{doc.page_content}")
        print(f"{'='*60}\n")

        context = "\n\n".join(context_parts)

        prompt_system = (
            "You are a board-certified cardiologist acting as a Clinical "
            "Decision Support Assistant. Your role is to help medical "
            "professionals interpret symptoms and clinical findings.\n\n"
            "RULES:\n"
            "1. Base your response EXCLUSIVELY on the provided Context. "
            "Do NOT introduce facts, statistics, drug names, dosages, or "
            "diagnostic criteria that are not EXPLICITLY written in the Context.\n"
            "2. Structure your response as follows:\n"
            "   **Clinical Summary** — a concise interpretation of the "
            "symptoms or clinical scenario described.\n"
            "   **Evidence-Based Insights** — key findings from the Context "
            "that are relevant to the query, with direct references to "
            "the text. Only include information you can point to in the Context.\n"
            "   **Limitations** — what the Context does NOT cover and what "
            "additional workup or specialist input may be needed.\n"
            "3. Always end with the disclaimer: 'This output is for "
            "informational use by medical professionals only and does not "
            "constitute a diagnosis or treatment recommendation.'\n\n"
            "<CRITICAL_RULE>\n"
            "Before answering, assess whether the provided Context directly "
            "addresses the specific condition or entity named in the query. "
            "If the Context discusses only related but distinct conditions, or "
            "if the specific diagnosis, treatment protocol, or clinical "
            "criteria needed to answer the query is NOT explicitly present "
            "in the Context, you MUST respond with: 'Insufficient evidence "
            "in the current knowledge base to address this specific query.' "
            "Do NOT fill gaps using your own medical training. "
            "Inventing information that is absent from the Context is "
            "strictly prohibited and risks patient harm.\n"
            "</CRITICAL_RULE>"
        )
        prompt_user = (
            f"Evidence Context:\n{context}\n\n"
            f"Clinical Query:\n{question}"
        )

        try:
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.0,
                max_tokens=1024,
                extra_headers={"x-folder-id": YANDEX_PROJECT_ID},
            )
            final_answer = response.choices[0].message.content.strip()

            evidence_md = ""
            if valid_docs_and_scores:
                for i, (doc, l2_score) in enumerate(valid_docs_and_scores, 1):
                    source = doc.metadata.get("source_file", "unknown")
                    category = doc.metadata.get("category", "unknown")
                    sim_score = max(0.0, 1.0 - (l2_score / MAX_L2_DISTANCE))
                    evidence_md += f"### Chunk {i} | Source: {source} | Category: {category} | Confidence: {sim_score:.1%}\n"
                    evidence_md += f"```text\n{doc.page_content}\n```\n\n"
            else:
                evidence_md = "No evidence retrieved."

            return final_answer, evidence_md
        except Exception as e:
            return (
                "The cardiology agent encountered an error while generating "
                f"a response. Please try again later. (Detail: {type(e).__name__}: {e})"
            ), "Error retrieving evidence."
