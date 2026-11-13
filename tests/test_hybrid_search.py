from search.hybrid_search import HybridSearch, normalize_scores, rrf_score


def test_normalize_scores():
    assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert normalize_scores([3.0, 3.0]) == [1.0, 1.0]
    assert normalize_scores([]) == []


def test_rrf_score_rewards_higher_ranks():
    assert rrf_score(1) > rrf_score(2) > rrf_score(10)


def test_rrf_search_finds_the_relevant_document(sample_data):
    from search.documents import load_documents

    hs = HybridSearch(load_documents())
    results = hs.rrf_search("bear who loves marmalade", limit=3)

    assert results[0]["document"]["title"] == "Paddington"
    assert results[0]["bm25_rank"] == 1


def test_results_always_contain_the_full_document(sample_data):
    from search.documents import load_documents

    documents = load_documents()
    hs = HybridSearch(documents)

    for entry in hs.rrf_search("space planet", limit=4):
        original = next(d for d in documents if d["id"] == entry["document"]["id"])
        assert entry["document"]["description"] == original["description"]


def test_index_is_rebuilt_for_a_different_dataset(sample_data, tmp_path, monkeypatch):
    import json

    from search.documents import load_documents

    HybridSearch(load_documents())  # builds and caches the index for the first dataset

    other = tmp_path / "other.json"
    other.write_text(json.dumps([{"title": "Volcano", "description": "Lava erupts from the mountain."}]))
    monkeypatch.setenv("RAG_DATA_PATH", str(other))

    hs = HybridSearch(load_documents())
    assert hs.rrf_search("lava mountain", limit=1)[0]["document"]["title"] == "Volcano"
