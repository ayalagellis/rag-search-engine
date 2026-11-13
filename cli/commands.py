import sys

from search.constants import BM25_B, BM25_K1
from search.inverted_index import InvertedIndex


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        return idx.get_bm25_idf(term)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        return idx.get_bm25_tf(doc_id, term, k1, b)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def bm25_search_command(query: str, limit: int = 5) -> list[tuple[dict, float]]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)

    return idx.bm25_search(query, limit)