from typing import List, Callable


def search_queries(queries: List, search_fn: Callable, **kwargs):
    for i, query in enumerate(queries, 1):
        print(50 * "-")
        print(f"Top 5 passages for query: '{query}'")
        print(50 * "-")

        results = search_fn(query, **kwargs)

        for j, result in enumerate(results, 1):
            print(f"\n{j}. Score: {result['score']:.4f}")
            print(f"Passage: {result['passage']}")

        print(50 * "-")
