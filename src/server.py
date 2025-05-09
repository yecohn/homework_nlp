import os
import nltk
import uvicorn
from fastapi import FastAPI, Body, Depends
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from lexical_retrieval import vectorize, preprocess_text, search
from semantic_search import semantic_search, split_into_chunks
from metrics import evaluate_rankings

TOP_K = 5


class ObjectLoader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True

    def load_objects(
        self, model, chunk_embeddings, chunks, vectorizer, passages, tfidf_matrix
    ):
        self.model = model
        self.chunk_embeddings = chunk_embeddings
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.passages = passages
        self.tfidf_matrix = tfidf_matrix


# init object loader to dump elements at server startup
loader = ObjectLoader()


# callback for getting loader object
def get_loader():
    return loader


app = FastAPI(
    title="Harry Potter Search API",
    description="API for semantic and lexical search in Harry Potter text",
)


# Define response models
class SearchResult(BaseModel):
    passage: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """add an on startup event to preload model and embeddings for faster inference when querying endpoint.

    Args:
        app: FastApi server
    """
    try:

        nltk.download("punkt")
        nltk.download("punkt_tab")
        # Load the text file
        with open("harry_potter.txt", "r", encoding="utf-8") as file:
            text = file.read()

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        chunks = split_into_chunks(text)
        chunk_embeddings = model.encode(chunks)
        vectorizer, passages = vectorize(text, TfidfVectorizer(), return_passages=True)
        passages = chunks  # Use same 3-sentence chunks for both
        tfidf_matrix = vectorizer.transform(passages)
        loader = ObjectLoader()
        loader.load_objects(
            model=model,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            vectorizer=vectorizer,
            passages=passages,
            tfidf_matrix=tfidf_matrix,
        )
        yield

    finally:
        print("shutting down server")


app = FastAPI(lifespan=lifespan)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Harry Potter Search API",
        "endpoints": {
            "/search/semantic": "Semantic search using sentence-transformers",
            "/search/tfidf": "TF-IDF search using sklearn",
            "/compare_methods": "Compare the results of the semantic and tfidf search methods",
        },
    }


@app.post(
    "/search/semantic",
)
async def semantic_search_endpoint(
    q: str = Body(..., description="Search query"),
    top_k: Optional[int] = Body(TOP_K, description="Number of results to return"),
    loader: ObjectLoader = Depends(get_loader),
):
    """
    Perform semantic search using sentence-transformers

    This endpoint uses the multi-qa-MiniLM-L6-cos-v1 model to find semantically
    similar passages in the Harry Potter text.
    """
    model = loader.model
    chunks = loader.chunks
    chunk_embeddings = loader.chunk_embeddings
    results = semantic_search(
        query=q,
        model=model,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=int(top_k),
    )

    # Convert results to the response model format
    return SearchResponse(
        query=q,
        results=[
            SearchResult(passage=r["passage"], score=float(r["score"])) for r in results
        ],
        search_type="semantic",
    )

@app.post(
    "/search/tfidf",
)
async def tfidf_endpoint(
    q: str = Body(..., description="Search query"),
    top_k: Optional[int] = Body(TOP_K, description="Number of results to return"),
    loader: ObjectLoader = Depends(get_loader),
):
    """
    Perform semantic search using sentence-transformers

    This endpoint uses the multi-qa-MiniLM-L6-cos-v1 model to find semantically
    similar passages in the Harry Potter text.
    """
    vectorizer = loader.vectorizer
    passages = loader.passages
    tfidf_matrix = loader.tfidf_matrix
    results = search(
        query=q,
        top_k=int(top_k),
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        passages=passages,
    )

    # Convert results to the response model format
    return SearchResponse(
        query=q,
        results=[
            SearchResult(passage=r["passage"], score=float(r["score"])) for r in results
        ],
        search_type="tfidf",
    )


@app.post("/compare_methods")
async def compare_method_endpoints(
    q: str = Body(..., description="Search query"),
    top_k: Optional[int] = Body(TOP_K, description="Number of results to return"),
    loader: ObjectLoader = Depends(get_loader),
):
    """
    Compare the results of the semantic and tfidf search methods
    """
    semantic_results = await semantic_search_endpoint(q, top_k, loader)
    semantic_results = semantic_results.results
    tfidf_results = await tfidf_endpoint(q, top_k, loader)
    tfidf_results = tfidf_results.results
    relevant_passages = set([result.passage for result in semantic_results])
    y_true = [1 if result.passage in relevant_passages else 0 for result in semantic_results]
    y_pred = [result.score for result in tfidf_results]
    # Evaluate using our metrics
    metrics = evaluate_rankings(y_true, y_pred, [top_k])

    return {
        "semantic_results": semantic_results,
        "tfidf_results": tfidf_results,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Run the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
