# ai_chatbot/rag/vector_store.py

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os
import uuid

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

bm25 = BM25Encoder().default()

def delete_document_vectors(namespace, filename):
    """
    Deletes all vectors in a namespace associated with a specific filename.
    Safe version: Queries existing exact vector IDs first and deletes them by ID.
    This prevents Pinecone's serverless eventual-consistency from accidentally wiping 
    new replacement chunks that share the same metadata filename during an immediate re-upload.
    """
    try:
        # Query up to 10k chunks associated with this exact filename
        results = index.query(
            vector=[0.0] * 768, # dummy vector for purely metadata-based retrieval
            filter={"filename": {"$eq": filename}},
            top_k=10000, 
            namespace=namespace,
            include_values=False,
            include_metadata=False
        )
        
        ids_to_delete = [match["id"] for match in results.get("matches", [])]
        
        if ids_to_delete:
            # Delete exact IDs synchronously 
            index.delete(ids=ids_to_delete, namespace=namespace)
            
    except Exception as e:
        print(f"Error querying/deleting existing documents: {e}")

def clear_namespace(namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        if "Namespace not found" in str(e):
            # Normal if this is the user's first document
            pass
        else:
            print(f"Error clearing namespace: {e}")


def upsert_chunks(chunks, embeddings, namespace, filename, custom_metadata=None):
    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = custom_metadata[i] if custom_metadata else {
            "text": chunk,
            "is_slide": False
        }
        
        # Add filename to metadata for proper querying and deletion later
        meta["filename"] = filename

        # Use an ID that includes a UUID to prevent Pinecone's eventual-consistency overwriting old deleted IDs
        vectors.append({
            "id": f"{namespace}_{filename}_{i}_{uuid.uuid4().hex[:8]}",
            "values": emb,
            "metadata": meta
        })

    # Batch upsert in chunks of 100 to prevent Pinecone payload size limit errors
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)


def index_embeddings(embeddings, namespace):
    """
    Indexes the given embeddings in Pinecone.
    """
    clear_namespace(namespace)
    upsert_chunks(embeddings["chunks"], embeddings["embeddings"], namespace)


def retrieve_context(query, namespace, top_k=12, score_threshold=0.25):
    """
    Retrieves the most relevant text chunks from Pinecone.
    """
    from .embedding import embed_query

    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    # Group matches by filename for annotation, but preserve order by score
    context_parts = []
    seen = set()
    for match in results["matches"]:
        # Filter chunks that fall below the relevance threshold
        if match.get("score", 0) < score_threshold:
            continue

        meta = match["metadata"]
        text = meta.get("text", "")
        filename = meta.get("filename", "unknown")
        doc_type = "Slide Deck" if meta.get("is_slide") else "PDF/Document"
        # Optionally include slide/page index if available
        index_info = ""
        if meta.get("is_slide") and meta.get("slide_index") is not None:
            index_info = f" (Slide {meta.get('slide_index')})"
        elif meta.get("page") is not None:
            index_info = f" (Page {meta.get('page')})"

        # Annotate each chunk with its document source for LLM synthesis
        annotated = f"[Source: {filename} | Type: {doc_type}{index_info}] {text}"
        # Avoid duplicate chunks (by id)
        chunk_id = match.get("id")
        if chunk_id and chunk_id in seen:
            continue
        seen.add(chunk_id)
        context_parts.append(annotated)

    # Merge all relevant context chunks from all documents
    return "\n\n".join(context_parts)