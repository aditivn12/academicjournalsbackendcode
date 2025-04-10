from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from utils.semanticsplitter import split_text_semantically
from utils.embeddings import make_embeddings
from utils.pinecone_ops import upsert_embeddings, query_similar_chunks

load_dotenv()
client = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        num_chunks = upsert_embeddings(chunks, embeddings)
        return UploadResponse(
            message="Article upserted and embedded successfully in Pinecone.",
            num_chunks=num_chunks
        )
    except Exception as e:
        print("Error in /upsert:", e)
        raise HTTPException(status_code=500, detail=f"Failed to process article: {str(e)}")
@app.post("/chat", response_model=ChatResponse)
def chat_with_article(data: QuestionInput) -> ChatResponse:
    try:
        print("Incoming question:", data.question)
        question_embedding = make_embeddings([data.question])[0]
        top_chunks = query_similar_chunks(question_embedding, top_k=3)
        if not top_chunks:
            return ChatResponse(response="No relevant information found. Please make sure you've uploaded an article first.")
        context = "\n\n".join(top_chunks)
        response = client.chat.completions.create(
            model="gpt-4o",
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
        print("Error in /chat:", e)
        traceback.print_exc()
        return ChatResponse(response=f"Sorry, something went wrong during processing: {str(e)}")