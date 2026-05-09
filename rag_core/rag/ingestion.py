# rag_core/rag/ingestion.py

from .pdf_loader import load_pdf
from .chunking import chunk_text
from .embedding import embed_texts
from .vector_store import upsert_chunks, clear_namespace


def process_pdf(file, session_id):
    # clear old document vectors from the user's namespace
    clear_namespace(session_id)

    # 1. Load
    text = load_pdf(file)

    # 2. Chunk
    chunks = chunk_text(text)

    # 3. Embed
    embeddings = embed_texts(chunks)

    # 4. Store (namespace = session)
    upsert_chunks(chunks, embeddings, namespace=session_id)

    return len(chunks)