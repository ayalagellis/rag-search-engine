import argparse

from search.hybrid_search import HybridSearch
from search.llm_client import generate
from search.semantic_search import load_movies


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results using LLM")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize")

    citations_parser = subparsers.add_parser("citations", help="Answer query with cited sources")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to use")

    question_parser = subparsers.add_parser("question", help="Answer a question conversationally using search results")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to use")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            titles = [entry["document"].get("title", "Unknown") for entry in results]
            docs = "\n".join(
                f"- {entry['document'].get('title', '')}: {entry['document'].get('description', '')[:200]}"
                for entry in results
            )

            prompt = f"""You are a search assistant that answers questions using the retrieved documents below.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            print("Search Results:")
            for title in titles:
                print(f"- {title}")
            print("\nRAG Response:")
            print(generate(prompt))

        case "summarize":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            titles = [entry["document"].get("title", "Unknown") for entry in results]
            results_str = "\n".join(
                f"- {entry['document'].get('title', '')}: {entry['document'].get('description', '')[:200]}"
                for entry in results
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with the key facts from each result.

This should help the user understand their options.

Query: {query}

Search results:
{results_str}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            print("\nLLM Summary:")
            print(generate(prompt))
            print()

        case "citations":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            titles = [entry["document"].get("title", "Unknown") for entry in results]
            documents_str = "\n".join(
                f"[{i}] {entry['document'].get('title', '')}: {entry['document'].get('description', '')[:200]}"
                for i, entry in enumerate(results, 1)
            )

            prompt = f"""Answer the query below and give information based on the provided documents.


If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents_str}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            print("\nLLM Answer:")
            print(generate(prompt))
            print()

        case "question":
            question = args.question
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(question, k=60, limit=args.limit)

            titles = [entry["document"].get("title", "Unknown") for entry in results]
            context = "\n".join(
                f"- {entry['document'].get('title', '')}: {entry['document'].get('description', '')[:200]}"
                for entry in results
            )

            prompt = f"""Answer the user's question based on the provided documents.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            print("Search Results:")
            for title in titles:
                print(f"  - {title}")
            print("\nAnswer:")
            print(generate(prompt))
            print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()