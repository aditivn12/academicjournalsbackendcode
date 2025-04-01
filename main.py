from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils.semanticsplitter import split_text_semantically
from utils.embeddings import make_embeddings
from utils.uproot import get_relevant_chunks
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.state.stored_chunks: List[str] = []
app.state.stored_embeddings: np.ndarray = np.array([])

class ArticleInput(BaseModel):
    article: str

class QuestionInput(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    num_chunks: int

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def root():
    return {"message": "Welcome to the Academic Article Chatbot API!"}

@app.post("/upsert", response_model=UploadResponse)
def upsert_article(data: ArticleInput) -> UploadResponse:
    print("Received article:", data.article)
    chunks: List[str] = split_text_semantically(data.article)
    print("Chunks created:", chunks)
    embeddings: np.ndarray = make_embeddings(chunks)
    print("Embeddings shape:", embeddings.shape)

    app.state.stored_chunks = chunks
    app.state.stored_embeddings = embeddings

    return UploadResponse(message="Article upserted and embedded successfully.", num_chunks=len(chunks))


@app.post("/chat", response_model=ChatResponse)
def chat_with_article(data: QuestionInput) -> ChatResponse:
    if app.state.stored_embeddings is None or len(app.state.stored_embeddings) == 0:
        return ChatResponse(response="No article uploaded yet. Please upsert an article first.")

    question_embedding = make_embeddings([data.question])[0]

    top_chunks: List[str] = get_relevant_chunks(
        query_vec=np.array([question_embedding]),
        doc_embeddings=app.state.stored_embeddings,
        doc_chunks=app.state.stored_chunks,
        k=3
    )

    context: str = "\n\n".join(top_chunks)
    

    fake_answer = f"Based on the article, here's what I found:\n\n{context}"

    return ChatResponse(response=fake_answer)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
