import os
from typing import List
import numpy as np
from pinecone import Pinecone, ServerlessSpec

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env file")
    return Pinecone(api_key=api_key)

def get_or_create_index(index_name="article-embeddings"):
    pc = get_pinecone_client()

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)

def upsert_embeddings(chunks: List[str], embeddings: np.ndarray, index_name="article-embeddings"):
    index = get_or_create_index(index_name)

    vectors = [
        {
            "id": f"chunk_{i}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    return len(vectors)

def query_similar_chunks(query_embedding: np.ndarray, top_k=3, index_name="article-embeddings"):
    index = get_or_create_index(index_name)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    matches = results.get("matches", [])
    if not matches:
        return []

    return [match["metadata"]["text"] for match in matches]
