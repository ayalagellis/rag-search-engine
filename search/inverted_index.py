import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

from search.config import cache_path
from search.constants import BM25_B, BM25_K1
from search.documents import load_documents

with open("data/stopwords.txt", "r") as f:
    STOPWORDS = set(f.read().splitlines())

_stemmer = PorterStemmer()
_translator = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(_translator)
    tokens = text.split()
    cleaned = []
    for t in tokens:
        if t and t not in STOPWORDS:
            cleaned.append(_stemmer.stem(t))
    return cleaned


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        token = term.lower()
        return sorted(self.index.get(token, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        if not tokens:
            return 0
        token = tokens[0]
        return self.term_frequencies.get(doc_id, Counter())[token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        token = tokens[0] if tokens else term.lower()
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        token = tokens[0] if tokens else term.lower()
        N = len(self.docmap)
        df = len(self.index.get(token, set()))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        query_tokens = tokenize(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            total = 0.0
            for token in query_tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        top_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:limit]
        return [(self.docmap[doc_id], scores[doc_id]) for doc_id in top_ids]

    def build(self, documents: list[dict] | None = None) -> None:
        if documents is None:
            documents = load_documents()

        for doc in documents:
            doc_id = doc["id"]
            self.docmap[doc_id] = doc
            text = f"{doc['title']} {doc['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        with open(cache_path("index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(cache_path("docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(cache_path("term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(cache_path("doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        paths = {
            "index": cache_path("index.pkl"),
            "docmap": cache_path("docmap.pkl"),
            "term_frequencies": cache_path("term_frequencies.pkl"),
            "doc_lengths": cache_path("doc_lengths.pkl"),
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Index files not found. Run `build` first to generate the cache."
            )
        with open(paths["index"], "rb") as f:
            self.index = pickle.load(f)
        with open(paths["docmap"], "rb") as f:
            self.docmap = pickle.load(f)
        with open(paths["term_frequencies"], "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(paths["doc_lengths"], "rb") as f:
            self.doc_lengths = pickle.load(f)
            