# RAG Search Engine

A search engine and retrieval-augmented generation (RAG) system built from scratch in Python. It combines keyword search (BM25) with semantic vector search, fuses the results with Reciprocal Rank Fusion, and passes the best matches to an LLM (Google Gemini) to generate answers that cite their sources.

The retrieval layer — inverted index, BM25 scoring, chunking, embedding search and hybrid fusion — is implemented by hand. On top of it, a **LangChain** pipeline wraps the engine as a standard retriever and composes the answer-generation chain, and a **FastAPI web app** provides a browser UI and a JSON API. It works with **any document collection**: a JSON dataset or a folder of your own `.txt`, `.md` and `.pdf` files.

## Features

- **Bring your own documents:** point the engine at a JSON file or a folder of text, Markdown and PDF files. Each dataset gets its own cache, so you can switch between them freely.
- **Keyword search (BM25):** custom inverted index with tokenization, stop-word removal, Porter stemming, TF-IDF and BM25 scoring.
- **Semantic search:** sentence embeddings (`all-MiniLM-L6-v2`), sentence-based chunking with overlap, and vectorized cosine similarity in NumPy.
- **Hybrid search:** weighted score fusion (min-max normalized, tunable `alpha`) or **Reciprocal Rank Fusion (RRF)**.
- **LangChain RAG pipeline:** a custom `BaseRetriever` over the hybrid engine, composed with a prompt template, Gemini chat model and output parser into one chain that returns the answer together with its numbered sources.
- **Web UI and REST API:** ask questions or run searches in the browser, with cited answers and per-source keyword/semantic ranks.
- **LLM query enhancement:** optional spell correction, query rewriting or query expansion before searching.
- **Re-ranking:** per-result LLM scoring, batch LLM ranking, or a cross-encoder (`ms-marco-TinyBERT-L2-v2`).
- **Multimodal search:** search with an image using CLIP (`clip-ViT-B-32`) embeddings, or rewrite a query from an image with Gemini.
- **Evaluation:** Precision@k, Recall@k and F1 against a hand-labeled golden dataset.
- **Tests:** a pytest suite that runs offline with a lightweight fake embedding model and a fake LLM.

## Architecture

```
                         ┌──────────────── HybridSearch ────────────────┐
question ─► (optional) ─►│  BM25 keyword search ──┐                      │
            query        │                        ├─► Reciprocal Rank ───┼─► top-k documents
            enhancement  │  semantic chunk search ┘     Fusion           │         │
                         └──────────────────────────────────────────────┘         │
                                                                                   ▼
             LangChain chain:  HybridRetriever ─► prompt template ─► Gemini ─► answer + cited sources
                                                                                   │
                                                          CLI  ◄───────────────────┼───────────► FastAPI + web UI
```

## Tech stack

- **Language:** Python 3.14
- **Retrieval:** NumPy, NLTK, sentence-transformers (MiniLM, CLIP, cross-encoder)
- **LLM orchestration:** LangChain (`langchain-core`, `langchain-google-genai`), Google Gemini
- **Web:** FastAPI, Uvicorn, vanilla JavaScript
- **Documents:** pypdf for PDF text extraction
- **Tooling:** uv, pytest, python-dotenv

## Project structure

```
search/
  config.py           dataset path and per-dataset cache folders
  documents.py        loads JSON files or folders of .txt/.md/.pdf into one format
  inverted_index.py   inverted index, TF-IDF and BM25
  semantic_search.py  embeddings, chunking and vector search
  hybrid_search.py    weighted fusion and Reciprocal Rank Fusion
  langchain_rag.py    LangChain retriever and RAG chain
  llm_client.py       Gemini client
cli/                  command-line tools for each stage
web/                  FastAPI app and browser UI
data/sample_docs/     small example dataset (computer science notes)
tests/                pytest suite
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then clone and install:
   ```bash
   git clone https://github.com/ayalagellis/rag-search-engine.git
   cd rag-search-engine
   uv sync
   ```
2. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   RAG_DATA_PATH=data/sample_docs
   # Optional: pin a specific model, e.g. GEMINI_MODEL=gemini-3.8-flash
   ```
   `RAG_DATA_PATH` can be a JSON file or any folder of `.txt`, `.md` and `.pdf` files. JSON datasets can be a list of objects or an object with a `documents` list; each item needs a `title` and a text field (`description`, `text` or `content`).

The first run builds the index and embeddings for the dataset and caches them under `cache/<dataset name>/`; later runs load them from disk.

## Usage

### Web app

```bash
uv run uvicorn web.app:app --reload
```

Open http://127.0.0.1:8000 to ask questions or search. The same features are available as an API:

| Method | Endpoint      | Body                               | Returns                         |
|--------|---------------|------------------------------------|---------------------------------|
| GET    | `/api/info`   | –                                  | dataset name and document count |
| POST   | `/api/search` | `{"query": "...", "limit": 10}`    | ranked results with scores      |
| POST   | `/api/ask`    | `{"question": "..."}`              | answer and cited sources        |

### Command line

Run from the project root.

```bash
# LangChain RAG over any dataset
uv run python -m cli.langchain_rag_cli "What causes a page fault?" --data data/sample_docs

# Keyword search
uv run python -m cli.keyword_search_cli build
uv run python -m cli.keyword_search_cli bm25search "round robin scheduling"

# Semantic search over chunked embeddings
uv run python -m cli.semantic_search_cli search_chunked "how do programs get analyzed without source code"

# Hybrid search with query rewriting and cross-encoder re-ranking
uv run python -m cli.hybrid_search_cli rrf-search "memory paging" --enhance rewrite --rerank-method cross_encoder

# Answer, summarize, or cite sources
uv run python -m cli.augmented_generation_cli citations "What is hybrid search?"

# Search with an image
uv run python -m cli.multimodal_search_cli image_search path/to/image.jpg

# Evaluate retrieval quality (golden dataset labeled for the movies dataset)
uv run python -m cli.evaluation_cli --limit 5
```

### Tests

```bash
uv run pytest
```

The tests use a small in-memory dataset, a hashing-based fake embedding model and a fake LLM, so they run offline in a couple of seconds without an API key.
