from fastapi import FastAPI
from pydantic import BaseModel
from utils.split_string import split_string
from utils.embeddings import make_embeddings
from utils.uproot import get_relevant_chunks
from utils import memory
import numpy as np

app = FastAPI()

class ArticleInput(BaseModel):
    article: str

class QuestionInput(BaseModel):
    question: str

@app.post("/upload/")
def upload_article(data: ArticleInput):
    chunks = split_string(data.article)


    embeddings = make_embeddings(chunks)

    memory.stored_chunks = chunks
    memory.stored_embeddings = embeddings

    return {"message": "Article uploaded and embedded successfully.", "num_chunks": len(chunks)}

@app.post("/chat/")
def chat_with_article(data: QuestionInput):
    if not memory.stored_embeddings:
        return {"error": "No article uploaded yet. Please upload an article first."}


    question_embedding = make_embeddings([data.question])[0]

    top_chunks = get_relevant_chunks(
        query_vec=np.array([question_embedding]),
        doc_embeddings=memory.stored_embeddings,
        doc_chunks=memory.stored_chunks,
        k=3
    )


    context = "\n\n".join(top_chunks)
    fake_answer = f"Based on the article, here's what I found:\n\n{context}"

    return {"response": fake_answer}
