"""Shared test setup.

Tests use a tiny hashing "embedding model" instead of downloading a real
sentence-transformers model, and write all caches to a temporary folder,
so the suite runs offline in a few seconds.
"""

import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # stop words are loaded from data/stopwords.txt


class FakeEmbeddingModel:
    """Bag-of-words hashing embeddings: texts sharing words get similar vectors."""

    dimensions = 128
    max_seq_length = 256

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False):
        vectors = np.zeros((len(texts), self.dimensions))
        for row, text in enumerate(texts):
            for word in str(text).lower().split():
                word = word.strip(".,!?:;\"'()")
                if word:
                    digest = hashlib.md5(word.encode()).digest()
                    vectors[row, int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
        return vectors


try:
    import sentence_transformers  # noqa: F401
except ImportError:  # allow the tests to run without the heavy ML dependencies installed
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeEmbeddingModel
    stub.CrossEncoder = FakeEmbeddingModel
    sys.modules["sentence_transformers"] = stub


SAMPLE_DOCUMENTS = [
    {"id": 1, "title": "Paddington", "description": "A polite bear from Peru moves to London. He loves marmalade sandwiches and gets into trouble."},
    {"id": 2, "title": "The Martian", "description": "An astronaut is stranded on Mars. He grows potatoes to survive until a rescue mission arrives."},
    {"id": 3, "title": "Finding Nemo", "description": "A clownfish father searches the ocean for his lost son. A forgetful fish helps him on the journey."},
    {"id": 4, "title": "Interstellar", "description": "Explorers travel through a wormhole in space. They search for a new planet for humanity."},
]


@pytest.fixture
def sample_data(tmp_path, monkeypatch):
    """Point the project at a small temporary dataset with its own cache folder."""
    import search.config as config
    import search.semantic_search as semantic_search

    data_file = tmp_path / "sample.json"
    data_file.write_text(json.dumps({"documents": SAMPLE_DOCUMENTS}))

    monkeypatch.setenv("RAG_DATA_PATH", str(data_file))
    monkeypatch.setattr(config, "CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(semantic_search, "SentenceTransformer", FakeEmbeddingModel)
    return data_file
