import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture
def client(sample_data, monkeypatch):
    monkeypatch.setenv("RAG_PRELOAD", "0")
    from web import app as web_app
    from search.langchain_rag import create_rag_chain

    web_app.state.clear()
    web_app.state["chain"] = create_rag_chain(
        web_app.get_search(), k=2, llm=FakeListChatModel(responses=["A bear from Peru [1]."])
    )
    with TestClient(web_app.app) as test_client:
        yield test_client
    web_app.state.clear()


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Search Engine" in response.text


def test_info(client):
    assert client.get("/api/info").json() == {"dataset": "sample", "documents": 4}


def test_search(client):
    data = client.post("/api/search", json={"query": "marmalade bear", "limit": 2}).json()
    assert data["results"][0]["title"] == "Paddington"
    assert len(data["results"]) <= 2


def test_ask(client):
    data = client.post("/api/ask", json={"question": "Who is Paddington?"}).json()
    assert data["answer"] == "A bear from Peru [1]."
    assert data["sources"][0]["title"] == "Paddington"


def test_rejects_empty_query(client):
    assert client.post("/api/search", json={"query": ""}).status_code == 422
