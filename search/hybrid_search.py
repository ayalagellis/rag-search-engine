import os

from search.config import cache_path
from search.inverted_index import InvertedIndex
from search.semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.doc_by_id = {doc["id"]: doc for doc in documents}

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if os.path.exists(cache_path("index.pkl")):
            self.idx.load()
        if len(self.idx.docmap) != len(documents):
            self.idx = InvertedIndex()
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        # Drop documents that share no terms with the query so they don't get a rank.
        return [(doc, score) for doc, score in self.idx.bm25_search(query, limit) if score > 0]

    def _full_document(self, doc_id, fallback: dict) -> dict:
        # Semantic results only carry a short snippet, so always return the full document.
        return self.doc_by_id.get(doc_id, fallback)

    def weighted_search(self, query, alpha=0.5, limit=5):
        fetch_limit = min(limit * 10, 200)

        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]

        bm25_norm = normalize_scores(bm25_scores)
        semantic_norm = normalize_scores(semantic_scores)

        combined: dict[int, dict] = {}

        for (doc, _), norm in zip(bm25_results, bm25_norm):
            doc_id = doc["id"]
            combined[doc_id] = {
                "document": self._full_document(doc_id, doc),
                "bm25_score": norm,
                "semantic_score": 0.0,
            }

        for result, norm in zip(semantic_results, semantic_norm):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "document": self._full_document(doc_id, result),
                    "bm25_score": 0.0,
                    "semantic_score": norm,
                }
            else:
                combined[doc_id]["semantic_score"] = norm

        for entry in combined.values():
            entry["hybrid_score"] = hybrid_score(entry["bm25_score"], entry["semantic_score"], alpha)

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        fetch_limit = min(limit * 10, 200)

        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        combined: dict[int, dict] = {}

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc["id"]
            combined[doc_id] = {
                "document": self._full_document(doc_id, doc),
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "document": self._full_document(doc_id, result),
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf_score"] += rrf_score(rank, k)

        return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]
