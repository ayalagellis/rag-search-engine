import argparse
import logging
import sys

from google import genai

from search.hybrid_search import normalize_scores
from search.llm_client import DEFAULT_MODEL, SAFETY_SETTINGS, get_gemini_client


logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def enhance_query(query: str, method: str) -> str:
    client = get_gemini_client()

    if method == "spell":
        prompt = (
            f'Fix any spelling errors in the user-provided movie search query below.\n'
            f'Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.\n'
            f'Preserve punctuation and capitalization unless a change is required for a typo fix.\n'
            f'If there are no spelling errors, or if you\'re unsure, output the original query unchanged.\n'
            f'Output only the final query text, nothing else.\n'
            f'User query: "{query}"\n'
        )
        response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
        return response.text.strip()

    if method == "rewrite":
        prompt = (
            f'Rewrite the user-provided movie search query below to be more specific and searchable.\n\n'
            f'Consider:\n'
            f'- Common movie knowledge (famous actors, popular films)\n'
            f'- Genre conventions (horror = scary, animation = cartoon)\n'
            f'- Keep the rewritten query concise (under 10 words)\n'
            f'- It should be a Google-style search query, specific enough to yield relevant results\n'
            f'- Don\'t use boolean logic\n\n'
            f'Examples:\n'
            f'- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"\n'
            f'- "movie about bear in london with marmalade" -> "Paddington London marmalade"\n'
            f'- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"\n\n'
            f'If you cannot improve the query, output the original unchanged.\n'
            f'Output only the rewritten query text, nothing else.\n\n'
            f'User query: "{query}"\n'
        )
        response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
        return response.text.strip()

    if method == "expand":
        prompt = (
            f'Expand the user-provided movie search query below with related terms.\n\n'
            f'Add synonyms and related concepts that might appear in movie descriptions.\n'
            f'Keep expansions relevant and focused.\n'
            f'Output only the additional terms; they will be appended to the original query.\n\n'
            f'Examples:\n'
            f'- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"\n'
            f'- "action movie with bear" -> "action thriller bear chase fight adventure"\n'
            f'- "comedy with bear" -> "comedy funny bear humor lighthearted"\n\n'
            f'User query: "{query}"\n'
        )
        response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
        extra_terms = response.text.strip()
        return f"{query} {extra_terms}"

    return query


def rerank_results(query: str, results: list[dict], method: str) -> list[dict]:
    import time
    from google.genai import types

    client = get_gemini_client()
    print(f"Re-ranking top {len(results)} results using {method} method...")



    for entry in results:
        doc = entry["document"]
        prompt = (
            f'''Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")[:300]}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:'''
        )
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
        )
        try:
            entry["rerank_score"] = float(response.text.strip())
        except (ValueError, AttributeError):
            entry["rerank_score"] = 0.0
        time.sleep(3)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def rerank_results_batch(query: str, results: list[dict]) -> list[dict]:
    import json
    from google.genai import types

    client = get_gemini_client()
    print(f"Re-ranking top {len(results)} results using batch method...")



    doc_lines = []
    for entry in results:
        doc = entry["document"]
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        description = doc.get("description", "")[:100]
        doc_lines.append(f"ID {doc_id}: {title} - {description}")
    doc_list_str = "\n".join(doc_lines)

    prompt = (
        f'''Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:'''
    )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )

    try:
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        ranked_ids = json.loads(text)
    except (ValueError, AttributeError, Exception):
        return results

    id_to_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(ranked_ids)}
    for entry in results:
        doc_id = entry["document"].get("id")
        entry["rerank_rank"] = id_to_rank.get(doc_id, len(results) + 1)

    return sorted(results, key=lambda x: x.get("rerank_rank", len(results) + 1))



def rerank_results_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    from sentence_transformers import CrossEncoder

    print(f"Re-ranking top {len(results)} results using cross_encoder method...")

    pairs = []
    for entry in results:
        doc = entry["document"]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')[:500]}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for entry, score in zip(results, scores):
        entry["cross_encoder_score"] = float(score)

    return sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search with weighted alpha scoring")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 vs semantic (0=semantic, 1=BM25)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate results using LLM relevance scoring",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to apply after RRF search",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            from search.hybrid_search import HybridSearch
            from search.semantic_search import load_movies
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)[:args.limit]
            for i, entry in enumerate(results, 1):
                doc = entry["document"]
                title = doc.get("title", "Unknown")
                description = doc.get("description", "")[:100]
                print(f"{i}. {title}")
                print(f"  Hybrid Score: {entry['hybrid_score']:.3f}")
                print(f"  BM25: {entry['bm25_score']:.3f}, Semantic: {entry['semantic_score']:.3f}")
                print(f"  {description}...")

        case "rrf-search":
            from search.hybrid_search import HybridSearch
            from search.semantic_search import load_movies

            query = args.query
            logger.debug(f"Original query: '{query}'")

            if args.enhance:
                enhanced_query = enhance_query(query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
            logger.debug(f"Query after enhancement: '{query}'")

            fetch_limit = args.limit * 5 if args.rerank_method in ("individual", "cross_encoder") else min(args.limit * 3, 20) if args.rerank_method == "batch" else args.limit
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, args.k, fetch_limit)
            logger.debug(f"RRF results ({len(results)} total): {[e['document'].get('title') for e in results[:5]]}{'...' if len(results) > 5 else ''}")

            if args.rerank_method == "individual":
                results = rerank_results(query, results, args.rerank_method)
            elif args.rerank_method == "batch":
                results = rerank_results_batch(query, results)
            elif args.rerank_method == "cross_encoder":
                results = rerank_results_cross_encoder(query, results)

            logger.debug(f"Final results after re-ranking ({min(len(results), args.limit)} shown): {[e['document'].get('title') for e in results[:args.limit]]}")
            results = results[:args.limit]
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, entry in enumerate(results, 1):
                doc = entry["document"]
                title = doc.get("title", "Unknown")
                description = doc.get("description", "")[:100]
                bm25_rank = entry["bm25_rank"] if entry["bm25_rank"] is not None else "N/A"
                semantic_rank = entry["semantic_rank"] if entry["semantic_rank"] is not None else "N/A"
                print(f"{i}. {title}")
                if args.rerank_method == "individual":
                    print(f"   Re-rank Score: {entry.get('rerank_score', 0):.3f}/10")
                elif args.rerank_method == "batch":
                    print(f"   Re-rank Rank: {entry.get('rerank_rank', 'N/A')}")
                elif args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {entry.get('cross_encoder_score', 0):.3f}")
                print(f"   RRF Score: {entry['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {description}...")
                print()

            if args.evaluate:
                import json as _json
                from google.genai import types

                formatted_results = [
                    f"{i}. {entry['document'].get('title', '')} - {entry['document'].get('description', '')[:80]}"
                    for i, entry in enumerate(results, 1)
                ]
                eval_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

                eval_client = get_gemini_client()
                eval_response = eval_client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
                )
                try:
                    text = eval_response.text.strip()
                    if text.startswith("```"):
                        text = "\n".join(text.split("\n")[1:-1])
                    scores = _json.loads(text)
                except Exception:
                    scores = []

                print()
                for i, (entry, score) in enumerate(zip(results, scores), 1):
                    title = entry["document"].get("title", "Unknown")
                    print(f"{i}. {title}: {score}/3")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()