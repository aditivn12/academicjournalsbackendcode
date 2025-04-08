import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def make_embeddings(chunks: list[str]) -> np.ndarray:
    """Create embeddings using OpenAI's text-embedding-ada-002 model"""
    embeddings = []
    

    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        
        for embedding_data in response.data:
            embeddings.append(embedding_data.embedding)
    
    return np.array(embeddings)