# rag_core/rag/vector_store.py

from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def clear_namespace(namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        if "Namespace not found" in str(e):
            # Normal if this is the user's first document
            pass
        else:
            print(f"Error clearing namespace: {e}")


def upsert_chunks(chunks, embeddings, namespace):
    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{namespace}_{i}",
            "values": emb,
            "metadata": {
                "text": chunk
            }
        })

    # Batch upsert in chunks of 100 to prevent Pinecone payload size limit errors
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)