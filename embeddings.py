# ============================================================
# embeddings.py - Updated to use Google Gemini
# ============================================================

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
import os

def create_vector_store(chunks: List[Document]):
    print("\n🔢 Generating embeddings and building FAISS vector store...")
    print("   (This may take a moment depending on document size...)")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print(f"✅ Vector store created with {len(chunks)} embedded chunks.")
    return vector_store


def save_vector_store(vector_store, save_path: str = "faiss_index"):
    vector_store.save_local(save_path)
    print(f"💾 Vector store saved to './{save_path}/'")


def load_vector_store(load_path: str = "faiss_index"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved index found at: {load_path}")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = FAISS.load_local(
        load_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"📂 Vector store loaded from './{load_path}/'")
    return vector_store