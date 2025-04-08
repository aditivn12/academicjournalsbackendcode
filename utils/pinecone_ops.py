import os
import time
from typing import List
import numpy as np
from pinecone import Pinecone, ServerlessSpec

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env file")
    return Pinecone(api_key=api_key)

def get_or_create_index(index_name="article-embeddings", dimension=1536):
    pc = get_pinecone_client()

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


        while True:
            description = pc.describe_index(index_name)
            if description.status['ready']:
                break
            time.sleep(2)

    return pc.Index(index_name)

def upsert_embeddings(chunks: List[str], embeddings: np.ndarray, index_name="article-embeddings"):
    index = get_or_create_index(index_name)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    print(f"✅ Upserted {len(vectors)} vectors to Pinecone.")
    return len(vectors)

def query_similar_chunks(query_embedding: np.ndarray, top_k=3, index_name="article-embeddings"):
    index = get_or_create_index(index_name)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]
    if not matches:
        print("⚠️ No matches found.")
        return []

    for match in matches:
        print(f"🧠 Match score: {match['score']:.4f}")
        print(f"📄 Chunk: {match['metadata']['text'][:200]}...\n")

    return [match["metadata"]["text"] for match in matches]
