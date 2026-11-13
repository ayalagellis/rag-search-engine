"""Central configuration for the dataset and cache locations.

The dataset can be changed without touching the code by setting the
RAG_DATA_PATH environment variable (in .env or the shell). Each dataset
gets its own cache folder, so switching datasets never mixes up indexes.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_PATH = "data/movies.json"
CACHE_ROOT = "cache"


def data_path() -> str:
    return os.environ.get("RAG_DATA_PATH", DEFAULT_DATA_PATH)


def set_data_path(path: str) -> None:
    os.environ["RAG_DATA_PATH"] = path


def dataset_name() -> str:
    path = os.path.normpath(data_path())
    name = os.path.splitext(os.path.basename(path))[0] or "dataset"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def cache_dir() -> str:
    path = os.path.join(CACHE_ROOT, dataset_name())
    os.makedirs(path, exist_ok=True)
    return path


def cache_path(filename: str) -> str:
    return os.path.join(cache_dir(), filename)
