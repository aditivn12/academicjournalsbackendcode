from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def make_embeddings(chunks: list[str]) -> np.ndarray:
    return model.encode(chunks)
