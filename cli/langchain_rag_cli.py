import argparse

from search.config import dataset_name, set_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions with the LangChain RAG pipeline")
    parser.add_argument("question", type=str, help="Question to answer")
    parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--data", type=str, help="Dataset path (JSON file or folder of .txt/.md/.pdf files)")
    args = parser.parse_args()

    if args.data:
        set_data_path(args.data)

    from search.documents import load_documents
    from search.hybrid_search import HybridSearch
    from search.langchain_rag import create_rag_chain

    documents = load_documents()
    print(f"Dataset: {dataset_name()} ({len(documents)} documents)\n")

    chain = create_rag_chain(HybridSearch(documents), k=args.limit)
    result = chain.invoke(args.question)

    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"  [{i}] {doc.metadata['title']} (RRF score: {doc.metadata['rrf_score']})")


if __name__ == "__main__":
    main()
