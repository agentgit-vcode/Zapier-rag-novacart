"""Pinecone vector store — embedding, indexing, and search."""

import os

from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# --- Clients ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "novacart-rag")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_index():
    """Return a handle to the Pinecone index."""
    return pc.Index(INDEX_NAME)


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a text string."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a batch of text strings."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def upsert_chunks(chunks: list[dict], batch_size: int = 50):
    """Upsert document chunks into Pinecone.

    Each chunk dict must have:
        - chunk_id: str
        - text: str
        - metadata: dict (file_name, file_type, etc.)
    """
    index = get_index()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            metadata = chunk.get("metadata", {})
            metadata["chunk_text"] = chunk["text"][:1000]  # store preview in metadata
            vectors.append({
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": metadata,
            })

        index.upsert(vectors=vectors)

    return len(chunks)


def search_chunks(query: str, top_k: int = 8) -> list[dict]:
    """Search Pinecone for chunks relevant to the query."""
    index = get_index()
    query_embedding = embed_text(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        chunks.append({
            "chunk_id": match["id"],
            "score": match["score"],
            "text": metadata.get("chunk_text", ""),
            "file_type": metadata.get("file_type", "unknown"),
            "metadata": metadata,
        })

    return chunks


def get_index_stats() -> dict:
    """Return basic stats about the Pinecone index."""
    index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension", EMBEDDING_DIM),
    }
