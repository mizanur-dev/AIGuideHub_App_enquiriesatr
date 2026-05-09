# rag_core/rag/embedding.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def embed_texts(texts):
    embeddings_model = get_embedding_model()
    return embeddings_model.embed_documents(texts)
