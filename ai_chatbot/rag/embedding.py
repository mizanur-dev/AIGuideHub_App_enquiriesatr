# ai_chatbot/rag/embedding.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class SafeGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Temporary Fix: The latest `langchain-google-genai` SDK drops batch responses 
    and returns a single embedding when using `gemini-embedding-2`. 
    This custom wrapper loops embed_query to ensure list consistency.
    """
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

def get_embedding_model():
    return SafeGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=768
    )

def embed_texts(texts):
    embeddings_model = get_embedding_model()
    return embeddings_model.embed_documents(texts)

def embed_query(query):
    embeddings_model = get_embedding_model()
    return embeddings_model.embed_query(query)
