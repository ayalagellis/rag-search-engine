import json

import pytest

from search.documents import load_documents


def test_loads_json_object_with_documents_key(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps({"documents": [{"title": "A", "text": "First document."}]}))

    docs = load_documents(str(path))

    assert docs == [{"title": "A", "text": "First document.", "id": 1, "description": "First document."}]


def test_loads_json_list_and_keeps_existing_ids(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"id": 42, "name": "Named", "content": "Some content."}]))

    docs = load_documents(str(path))

    assert docs[0]["id"] == 42
    assert docs[0]["title"] == "Named"
    assert docs[0]["description"] == "Some content."


def test_loads_folder_of_text_files(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "operating_systems.md").write_text("# Scheduling\nRound robin gives each process a time slice.")
    (tmp_path / "networks.txt").write_text("TCP guarantees ordered delivery.")
    (tmp_path / "ignored.csv").write_text("not,loaded")

    docs = load_documents(str(tmp_path))

    assert [d["title"] for d in docs] == ["networks", "operating systems"]
    assert [d["id"] for d in docs] == [1, 2]
    assert "Round robin" in docs[1]["description"]
    assert docs[1]["source"].endswith("operating_systems.md")


def test_skips_documents_without_text(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"title": "Empty", "description": "  "}, {"title": "Full", "description": "Text."}]))

    assert [d["title"] for d in load_documents(str(path))] == ["Full"]


def test_missing_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="RAG_DATA_PATH"):
        load_documents(str(tmp_path / "missing.json"))
