# ai_chatbot/rag/vector_store.py

import logging
import os
import uuid

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

logger = logging.getLogger(__name__)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

bm25 = BM25Encoder().default()


def _sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [str(item) for item in value]
        else:
            sanitized[key] = str(value)
    return sanitized


def delete_document_vectors(namespace, filename, module_id=None):
    """
    Deletes all vectors in a namespace associated with a specific filename,
    and optionally within a module.
    """
    filter_dict = {"filename": {"$eq": filename}}
    if module_id is not None:
        filter_dict["module_id"] = {"$eq": module_id}

    try:
        index.delete(filter=filter_dict, namespace=namespace)
        return
    except Exception as e:
        if "Namespace not found" in str(e):
            logger.debug("Pinecone namespace %s not found; nothing to delete.", namespace)
            return
        logger.warning("Pinecone filter-only delete failed; falling back to query-based delete. %s", e)

    try:
        results = index.query(
            vector=[0.0] * 768,
            filter=filter_dict,
            top_k=10000,
            namespace=namespace,
            include_values=False,
            include_metadata=False
        )

        ids_to_delete = [match["id"] for match in results.get("matches", [])]
        if ids_to_delete:
            index.delete(ids=ids_to_delete, namespace=namespace)

    except Exception as e:
        if "Namespace not found" in str(e):
            logger.debug("Pinecone namespace %s not found during query delete; nothing to delete.", namespace)
            return
        logger.error("Error querying/deleting existing documents: %s", e)

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
        sanitized_meta = _sanitize_metadata(meta)

        # Use an ID that includes a UUID to prevent Pinecone's eventual-consistency overwriting old deleted IDs
        vectors.append({
            "id": f"{namespace}_{filename}_{i}_{uuid.uuid4().hex[:8]}",
            "values": emb,
            "metadata": sanitized_meta
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


def retrieve_context(query, namespace, top_k=12, score_threshold=0.25, filename=None, module_id=None, subsection_name=None):
    """
    Retrieves the most relevant text chunks from Pinecone, optionally scoped by module or subsection.
    """
    from .embedding import embed_query

    query_embedding = embed_query(query)

    filter_dict = {}
    if filename:
        filter_dict["filename"] = filename
    if module_id is not None:
        filter_dict["module_id"] = {"$eq": module_id}
    if subsection_name:
        filter_dict["subsection_name"] = {"$eq": subsection_name}

    kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": namespace,
    }
    if filter_dict:
        kwargs["filter"] = filter_dict

    results = index.query(**kwargs)

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
        module_meta = ""
        if meta.get("module_id") is not None:
            module_meta = f" | Module: {meta.get('module_name', meta.get('module_id'))}"
        subsection_meta = ""
        if meta.get("subsection_name"):
            subsection_meta = f" | Subsection: {meta.get('subsection_name')}"
        # Optionally include slide/page index if available
        index_info = ""
        if meta.get("is_slide") and meta.get("slide_index") is not None:
            index_info = f" (Slide {meta.get('slide_index')})"
        elif meta.get("page") is not None:
            index_info = f" (Page {meta.get('page')})"

        # Annotate each chunk with its document source for LLM synthesis
        annotated = f"[Source: {filename} | Type: {doc_type}{module_meta}{subsection_meta}{index_info}] {text}"
        # Avoid duplicate chunks (by id)
        chunk_id = match.get("id")
        if chunk_id and chunk_id in seen:
            continue
        seen.add(chunk_id)
        context_parts.append(annotated)

    # Merge all relevant context chunks from all documents
    return "\n\n".join(context_parts)