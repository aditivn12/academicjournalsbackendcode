import numpy as np
import faiss
from typing import List
def get_relevant_chunks(query_vec: np.ndarray, doc_embeddings: np.ndarray, doc_chunks: List[str], k: int = 3) -> List[str]:
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(doc_embeddings))
    _, I = index.search(query_vec, k)
    return [doc_chunks[i] for i in I[0]]
