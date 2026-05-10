# ai_chatbot/rag/chunking.py
from langchain_experimental.text_splitter import SemanticChunker
from .embedding import get_embedding_model

def chunk_text(text):
    """
    Advanced Semantic Chunking: Moves away from fixed-size text chunking.
    Uses the Gemini embedding model to analyze the text's meaning and 
    groups sentences together dynamically based on semantic similarity.
    """
    if not text or not text.strip():
        return []

    # Initialize the same embedding model we use for indexing
    embeddings = get_embedding_model()
    
    # SemanticChunker creates chunks dynamically by mathematical thought meaning
    # instead of hard character length boundaries like 1k or 500 characters
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    docs = text_splitter.create_documents([text])
    
    return [doc.page_content for doc in docs]