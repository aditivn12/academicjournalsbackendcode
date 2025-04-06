from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import nltk
from dotenv import load_dotenv
nltk.download('punkt_tab')
from openai import OpenAI
client = OpenAI()
from dotenv import load_dotenv
load_dotenv()

from utils.semanticsplitter import split_text_semantically
from utils.embeddings import make_embeddings
from utils.uproot import get_relevant_chunks




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        print("📩 Incoming question:", data.question)
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

        response = client.chat.completions.create(
            model="gpt-3.5",
            messages=[
                {
                    "role": "user",
                    "content": f"Based on this article, answer the following question:\n\n{data.question}\n\nArticle:\n{context}"
                }
            ]
        )

        answer = response.choices[0].message.content.strip()
        return ChatResponse(response=answer)

    except Exception as e:
        import traceback
        print("❌ Error in /chat:", e)
        traceback.print_exc()
        return ChatResponse(response="Sorry, something went wrong during processing.")
