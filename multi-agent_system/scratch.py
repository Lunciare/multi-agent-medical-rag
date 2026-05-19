import json
import os
import faiss
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
from embeddings import YandexNativeEmbeddings
embeddings = YandexNativeEmbeddings()

vectorstore = FAISS.load_local("../data/processed/cardiology/faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("tests/data/golden_dataset.json", "r") as f:
    data = json.load(f)

for case in data:
    if case["tier"] == 2 and case["expected_specialist"] == "cardiologist":
        docs = vectorstore.similarity_search_with_score(case["query"], k=5)
        valid_docs = [doc for doc, score in docs if score <= 1.2]
        
        retrieved_text = " ".join([doc.page_content for doc in valid_docs]).lower()
        keywords = [kw.lower() for kw in case["expected_keywords"]]
        hit = any(kw in retrieved_text for kw in keywords)
        
        if not hit:
            print(f"MISS: {case['id']}")
            print(f"Condition inferred: {case['query']}")
            print(f"Keywords missing: {keywords}")
            print("-" * 50)
