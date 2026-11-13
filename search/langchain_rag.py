"""RAG pipeline built with LangChain on top of the custom hybrid search engine.

The retrieval side stays hand-built (BM25 + semantic search + Reciprocal Rank
Fusion). LangChain is used to wrap it as a standard retriever and compose the
generation step as a chain:

    question ──► HybridRetriever ──► format sources ──► prompt ──► Gemini ──► answer
                      │                                                       │
                      └───────────────── sources returned with ───────────────┘
"""

import os
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from pydantic import ConfigDict

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the numbered sources provided.

Rules:
- Base every statement on the sources and cite them inline as [1], [2], etc.
- If the sources do not contain the answer, say "I don't have enough information to answer that." and stop.
- Be direct and concise. Do not mention these rules."""

USER_PROMPT = """Sources:
{context}

Question: {question}"""


class HybridRetriever(BaseRetriever):
    """LangChain retriever backed by the project's HybridSearch (BM25 + semantic, fused with RRF)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search: Any
    k: int = 5
    rrf_k: int = 60
    max_chars: int = 1500

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.search.rrf_search(query, k=self.rrf_k, limit=self.k)
        documents = []
        for entry in results:
            doc = entry["document"]
            documents.append(
                Document(
                    page_content=doc.get("description", "")[: self.max_chars],
                    metadata={
                        "id": doc.get("id"),
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "rrf_score": round(entry["rrf_score"], 4),
                        "bm25_rank": entry.get("bm25_rank"),
                        "semantic_rank": entry.get("semantic_rank"),
                    },
                )
            )
        return documents


def format_sources(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {doc.metadata.get('title', '')}\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel | Runnable) -> Runnable:
    """Return a chain that takes a question string and returns {"question", "sources", "answer"}."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])

    generate_answer = (
        {
            "context": lambda x: format_sources(x["sources"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        question=RunnablePassthrough(),
        sources=retriever,
    ).assign(answer=generate_answer)


def get_gemini_chat_model(temperature: float = 0.2) -> Runnable:
    """Gemini chat model that falls back to a lighter model when the main one is busy or unavailable."""
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    from search.llm_client import DEFAULT_MODEL, FALLBACK_MODEL

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    def make(model: str) -> ChatGoogleGenerativeAI:
        # Fail fast instead of hanging: one minute per request and at most 2 retries
        # (the library default is 6 retries with growing waits and no timeout).
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            timeout=60,
            max_retries=2,
        )

    primary = make(DEFAULT_MODEL)
    if not FALLBACK_MODEL or FALLBACK_MODEL == DEFAULT_MODEL:
        return primary
    return primary.with_fallbacks([make(FALLBACK_MODEL)])


def create_rag_chain(search, k: int = 5, llm: BaseChatModel | Runnable | None = None) -> Runnable:
    retriever = HybridRetriever(search=search, k=k)
    return build_rag_chain(retriever, llm or get_gemini_chat_model())


def sources_to_dicts(documents: list[Document]) -> list[dict]:
    return [
        {"number": i, "snippet": doc.page_content[:300], **doc.metadata}
        for i, doc in enumerate(documents, start=1)
    ]

