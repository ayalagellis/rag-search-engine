import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from search.documents import load_documents


def _cosine_similarity(vec1, vec2) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32", documents: list[dict] = None) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in self.documents
        ]
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = []

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        result = self.model.encode([image])
        return result[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            score = _cosine_similarity(image_embedding, text_embedding)
            doc = self.documents[i]
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str) -> None:
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    documents = load_documents()
    ms = MultimodalSearch(documents=documents)
    return ms.search_with_image(image_path)