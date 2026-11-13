
import argparse
import re

from search.semantic_search import embed_query_text, embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model is loaded correctly")

    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings are generated correctly")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text input")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Search documents by semantic meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=40, help="Number of words to overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to overlap")

    subparsers.add_parser("embed_chunks", help="Build or load chunked embeddings for all documents")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search documents using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            from search.semantic_search import SemanticSearch, load_movies
            ss = SemanticSearch()
            ss.load_or_create_embeddings(load_movies())
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]}...")
                print()

        case "chunk":
            words = args.text.split()
            chunks = []
            step = args.chunk_size - args.overlap
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i + args.chunk_size])
                chunks.append(chunk)
                i += step
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            from search.semantic_search import _semantic_chunk_text
            chunks = _semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            from search.semantic_search import ChunkedSemanticSearch, load_movies
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(load_movies())
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            from search.semantic_search import ChunkedSemanticSearch, load_movies
            documents = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()