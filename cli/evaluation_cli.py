import argparse
import json

from search.hybrid_search import HybridSearch
from search.semantic_search import load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_titles = set(test_case["relevant_docs"])

        results = hs.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [entry["document"].get("title", "") for entry in results]

        relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant_titles)
        total_retrieved = len(retrieved_titles)
        total_relevant = len(relevant_titles)
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()