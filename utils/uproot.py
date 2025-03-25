import numpy as np
import faiss

def get_relevant_chunks(query_vec, doc_embeddings, doc_chunks, k=3):
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(doc_embeddings))
    D, I = index.search(query_vec, k)
    return [doc_chunks[i] for i in I[0]]
