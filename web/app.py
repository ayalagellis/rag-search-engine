"""FastAPI web app: a browser UI plus a small JSON API over the search engine.

Run from the project root:
    uv run uvicorn web.app:app --reload
Then open http://127.0.0.1:8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from search.config import dataset_name

logger = logging.getLogger("uvicorn.error")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

state: dict = {}


def get_search():
    """Build the search engine once, on first use, and reuse it for every request."""
    if "search" not in state:
        from search.documents import load_documents
        from search.hybrid_search import HybridSearch

        documents = load_documents()
        state["documents"] = documents
        state["search"] = HybridSearch(documents)
    return state["search"]


def get_chain():
    if "chain" not in state:
        from search.langchain_rag import create_rag_chain

        state["chain"] = create_rag_chain(get_search(), k=5)
    return state["chain"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RAG_PRELOAD", "1") == "1":
        get_search()
    yield


app = FastAPI(title="RAG Search Engine", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/info")
def info():
    get_search()
    return {"dataset": dataset_name(), "documents": len(state["documents"])}


@app.post("/api/search")
def search(request: SearchRequest):
    results = get_search().rrf_search(request.query.strip(), k=60, limit=request.limit)
    return {
        "query": request.query,
        "results": [
            {
                "id": entry["document"].get("id"),
                "title": entry["document"].get("title", ""),
                "snippet": entry["document"].get("description", "")[:300],
                "rrf_score": round(entry["rrf_score"], 4),
                "bm25_rank": entry.get("bm25_rank"),
                "semantic_rank": entry.get("semantic_rank"),
            }
            for entry in results
        ],
    }


@app.post("/api/ask")
def ask(request: AskRequest):
    from search.langchain_rag import sources_to_dicts

    try:
        result = get_chain().invoke(request.question.strip())
    except Exception as e:
        logger.exception("Failed to answer question")
        message = str(e)
        if "503" in message or "UNAVAILABLE" in message or "429" in message or "RESOURCE_EXHAUSTED" in message:
            raise HTTPException(
                status_code=503,
                detail="The AI model is busy or rate-limited right now. Please try again in a minute.",
            )
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources_to_dicts(result["sources"]),
    }
