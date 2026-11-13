from langchain_core.language_models.fake_chat_models import FakeListChatModel

from search.langchain_rag import HybridRetriever, build_rag_chain, format_sources, sources_to_dicts


class RecordingFakeLLM(FakeListChatModel):
    """Fake chat model that remembers the prompt it was given."""

    last_prompt: str = ""

    def _call(self, messages, *args, **kwargs):
        self.last_prompt = "\n".join(m.content for m in messages)
        return super()._call(messages, *args, **kwargs)


def make_search(sample_data):
    from search.documents import load_documents
    from search.hybrid_search import HybridSearch

    return HybridSearch(load_documents())


def test_retriever_returns_langchain_documents(sample_data):
    retriever = HybridRetriever(search=make_search(sample_data), k=2)

    docs = retriever.invoke("astronaut stranded on mars")

    assert len(docs) == 2
    assert docs[0].metadata["title"] == "The Martian"
    assert "potatoes" in docs[0].page_content


def test_chain_returns_answer_with_sources(sample_data):
    llm = RecordingFakeLLM(responses=["He grows potatoes [1]."])
    chain = build_rag_chain(HybridRetriever(search=make_search(sample_data), k=2), llm)

    result = chain.invoke("How does the astronaut survive on Mars?")

    assert result["answer"] == "He grows potatoes [1]."
    assert result["question"] == "How does the astronaut survive on Mars?"
    assert result["sources"][0].metadata["title"] == "The Martian"
    assert "[1] The Martian" in llm.last_prompt
    assert "How does the astronaut survive on Mars?" in llm.last_prompt


def test_format_and_serialize_sources(sample_data):
    docs = HybridRetriever(search=make_search(sample_data), k=1).invoke("clownfish ocean")

    assert format_sources(docs).startswith("[1] Finding Nemo\n")
    assert sources_to_dicts(docs)[0]["number"] == 1
    assert sources_to_dicts(docs)[0]["title"] == "Finding Nemo"
