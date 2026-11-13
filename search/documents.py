"""Load documents from any supported source into one common format.

Every document becomes a dict with "id", "title" and "description", which is
the shape the keyword, semantic and hybrid search code expects.

Supported sources:
  * a JSON file: a list of objects, or an object holding the list under
    "documents", "movies" or "items". Each object needs a title and a text
    field ("description", "text", "content" or "body").
  * a folder of .txt, .md and .pdf files (searched recursively). The file
    name becomes the title and the file contents become the text.
  * a single .txt, .md or .pdf file.
"""

import json
import os

from search.config import data_path

TEXT_FIELDS = ("description", "text", "content", "body")
LIST_KEYS = ("documents", "movies", "items")
FILE_EXTENSIONS = (".txt", ".md", ".pdf")


def load_documents(path: str | None = None) -> list[dict]:
    path = path or data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Set RAG_DATA_PATH to a JSON file "
            f"or a folder of .txt/.md/.pdf files."
        )

    if os.path.isdir(path):
        documents = _load_folder(path)
    elif path.lower().endswith(".json"):
        documents = _load_json(path)
    elif path.lower().endswith(FILE_EXTENSIONS):
        documents = [_load_file(path, os.path.dirname(path))]
    else:
        raise ValueError(f"Unsupported dataset type: '{path}'")

    documents = [d for d in documents if d["description"].strip()]
    if not documents:
        raise ValueError(f"No documents with text were found in '{path}'.")
    return documents


def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = next((data[k] for k in LIST_KEYS if isinstance(data.get(k), list)), None)
        if items is None:
            raise ValueError(
                f"JSON object in '{path}' must hold a list under one of: {', '.join(LIST_KEYS)}"
            )
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported JSON structure in '{path}'")

    documents = []
    for i, item in enumerate(items, start=1):
        text = next((item[k] for k in TEXT_FIELDS if isinstance(item.get(k), str)), "")
        documents.append({
            **item,
            "id": item.get("id", i),
            "title": str(item.get("title") or item.get("name") or f"Document {i}"),
            "description": text,
        })
    return documents


def _load_folder(folder: str) -> list[dict]:
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(FILE_EXTENSIONS):
                paths.append(os.path.join(root, name))

    documents = [_load_file(p, folder) for p in sorted(paths)]
    for i, doc in enumerate(documents, start=1):
        doc["id"] = i
    return documents


def _load_file(path: str, base: str) -> dict:
    if path.lower().endswith(".pdf"):
        text = _read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").replace("-", " ")
    return {
        "id": 1,
        "title": title,
        "description": " ".join(text.split()),
        "source": os.path.relpath(path, base) if base else os.path.basename(path),
    }


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
