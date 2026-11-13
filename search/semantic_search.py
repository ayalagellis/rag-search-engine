import json
import re
import os

import numpy as np
from search.config import cache_path
from search.constants import SCORE_PRECISION
from search.documents import load_documents
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "document_embeddings.npy"


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = None
        self.documents = None
        self.document_map: dict = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text must not be empty or whitespace.")
        result = self.model.encode([text])
        return result[0]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            results.append((score, self.documents[i]))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in results[:limit]
        ]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        doc_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        np.save(cache_path(EMBEDDINGS_FILE), self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(cache_path(EMBEDDINGS_FILE)):
            self.embeddings = np.load(cache_path(EMBEDDINGS_FILE))
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _cosine_similarities(query_vec, matrix) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix, in one NumPy operation."""
    matrix = np.asarray(matrix)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1.0
    return (matrix @ query_vec) / norms


def _semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [sentences[0]]

    sentences = [s.strip() for s in sentences if s.strip()]

    step = max_chunk_size - overlap
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + max_chunk_size])
        chunks.append(chunk)
        i += step
    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description or not description.strip():
                continue
            chunks = _semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(cache_path("chunk_embeddings.npy"), self.chunk_embeddings)
        with open(cache_path("chunk_metadata.json"), "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks), "total_documents": len(documents)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        scores = _cosine_similarities(query_embedding, self.chunk_embeddings)
        chunk_scores = [
            {
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": float(score),
            }
            for meta, score in zip(self.chunk_metadata, scores)
        ]

        movie_scores: dict[int, dict] = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in movie_scores or cs["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = cs

        sorted_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for entry in top_movies:
            doc = self.documents[entry["movie_idx"]]
            description = doc.get("description", "")
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": description[:100],
                "score": round(entry["score"], SCORE_PRECISION),
                "metadata": entry or {},
            })

        return results

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_file = cache_path("chunk_embeddings.npy")
        metadata_file = cache_path("chunk_metadata.json")
        if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
            self.chunk_embeddings = np.load(embeddings_file)
            with open(metadata_file, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            if data.get("total_documents", len(documents)) == len(documents):
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def load_movies(path: str | None = None) -> list[dict]:
    """Load the configured dataset (kept under its original name for the CLIs)."""
    return load_documents(path)

def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_documents()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")