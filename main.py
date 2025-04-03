from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import openai
import os
import nltk
from dotenv import load_dotenv
nltk.download('punkt_tab')


from utils.semanticsplitter import split_text_semantically
from utils.embeddings import make_embeddings
from utils.uproot import get_relevant_chunks


load_dotenv()
nltk.download('punkt')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store
app.state.stored_chunks: List[str] = []
app.state.stored_embeddings: np.ndarray = np.array([])

# Data models
class ArticleInput(BaseModel):
    article: str

class QuestionInput(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    num_chunks: int

class ChatResponse(BaseModel):
    response: str
    

# Routes
@app.get("/")
def root():
    return {"message": "Welcome to the Academic Article Chatbot API!"}

@app.post("/upsert", response_model=UploadResponse)
def upsert_article(data: ArticleInput) -> UploadResponse:
    try:
        chunks = split_text_semantically(data.article)
        embeddings = make_embeddings(chunks)

        app.state.stored_chunks = chunks
        app.state.stored_embeddings = embeddings

        return UploadResponse(
            message="Article upserted and embedded successfully.",
            num_chunks=len(chunks)
        )
    except Exception as e:
        print("❌ Error in /upsert:", e)
        raise HTTPException(status_code=500, detail="Failed to process article.")

@app.post("/chat", response_model=ChatResponse)
def chat_with_article(data: QuestionInput) -> ChatResponse:
    try:
        if app.state.stored_embeddings is None or len(app.state.stored_embeddings) == 0:
            return ChatResponse(response="No article uploaded yet. Please upsert an article first.")

        question_embedding = make_embeddings([data.question])[0]
        top_chunks = get_relevant_chunks(
            query_vec=np.array([question_embedding]),
            doc_embeddings=app.state.stored_embeddings,
            doc_chunks=app.state.stored_chunks,
            k=3
        )
        context = "\n\n".join(top_chunks)
        print("Querying Open AI")
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user"},
                {"content": ""}
            ]
        )

        answer = completion.choices[0].message.content.strip()
        return ChatResponse(response=answer)

    except Exception as e:
        print("❌ Error in /chat:", e)
        return ChatResponse(response="Sorry, something went wrong during processing.")
