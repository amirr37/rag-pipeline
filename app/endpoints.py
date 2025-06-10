from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:63342",
    "http://127.0.0.1:63342",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 2


class RetrieveResponse(BaseModel):
    documents: List[str]


class GenerateRequest(BaseModel):
    query: str
    # context: str = ""
    top_k: int = 2



class GenerateResponse(BaseModel):
    answer: str


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_docs(request: RetrieveRequest):
    query = request.query
    from retrieval import get_top_k_chunks
    documents = get_top_k_chunks(query, top_k=request.top_k)
    return RetrieveResponse(documents=documents)


@app.post("/generate", response_model=GenerateResponse)
def generate_answer(request: GenerateRequest):
    query = request.query
    from retrieval import get_top_k_chunks
    documents = get_top_k_chunks(query, top_k=request.top_k)
    from generation import ask_together
    response = ask_together(query, documents)

    return GenerateResponse(answer=response)


