# Retrieval interface for Prompt Generator
# Flow: query_text (str) -> embed -> ChromaDB similarity search -> structured result
#
# Prompt Generator usage:
#   from retrieve import query_similar, query_all
#   results = query_similar("The storm caused flooding", collection="maven_ere_causal")
#   results = query_all("The storm caused flooding")  # search all collections

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")

_model = None
_client = None

OUTPUT_FIELDS = ["policy", "domain_event", "command", "bounded_context",
                 "aggregate", "source_phrase", "views", "user_roles", "process"]


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def list_collections() -> list[str]:
    """Return all collection names available in ChromaDB."""
    return [c.name for c in _get_client().list_collections()]


def _build_result(distance: float, document: str, metadata: dict, collection: str) -> dict:
    """
    Standardised output structure returned to Prompt Generator:
        input    — the sentence that was embedded (query anchor)
        distance — similarity score (lower = more similar)
        collection — which collection this came from
        output   — all DDD metadata fields; missing fields are null
    """
    return {
        "input": document,
        "distance": round(distance, 4),
        "collection": collection,
        "output": {field: metadata.get(field, None) for field in OUTPUT_FIELDS},
    }


def query_similar(
    query_text: str,
    collection: str = "maven_ere_causal",
    n_results: int = 3,
) -> list[dict]:
    """
    Embed query_text and return n_results most similar records from one collection.
    Raises ValueError for empty/whitespace-only query.
    Raises chromadb.errors.InvalidCollectionException if collection does not exist.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    embedding = _get_model().encode([query_text], show_progress_bar=False).tolist()
    col = _get_client().get_collection(name=collection)

    col_size = col.count()
    safe_n = min(n_results, col_size)
    if safe_n == 0:
        return []

    results = col.query(
        query_embeddings=embedding,
        n_results=safe_n,
        include=["documents", "metadatas", "distances"],
    )

    return [
        _build_result(
            results["distances"][0][i],
            results["documents"][0][i],
            results["metadatas"][0][i],
            collection,
        )
        for i in range(len(results["ids"][0]))
    ]


def query_all(
    query_text: str,
    n_results_per_collection: int = 3,
) -> list[dict]:
    """
    Search all available collections and return merged results sorted by distance.
    Useful when the Prompt Generator does not know which collection to target.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    all_results = []
    for col_name in list_collections():
        try:
            all_results.extend(
                query_similar(query_text, collection=col_name,
                              n_results=n_results_per_collection)
            )
        except Exception:
            continue

    all_results.sort(key=lambda r: r["distance"])
    return all_results


# ---------------------------------------------------------------------------
# Demo — run directly to see live input/output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Retrieval Demo ===")
    print(f"Available collections: {list_collections()}\n")

    test_queries = [
        # MAVEN-ERE style
        ("maven_ere_causal", "The storm caused severe flooding in the region."),
        ("maven_ere_causal", "Political instability led to economic collapse."),
        # BPC style (will work once bpc_* collections are seeded)
        # ("bpc_logistics",    "Inventory shortages led to production delays."),
    ]

    for collection, query in test_queries:
        print(f"Query      : \"{query}\"")
        print(f"Collection : {collection}")
        results = query_similar(query, collection=collection, n_results=3)
        for rank, r in enumerate(results, 1):
            print(f"  Rank {rank}  distance={r['distance']}  collection={r['collection']}")
            print(f"    input  : {r['input'][:100]}")
            print(f"    output : {json.dumps(r['output'], ensure_ascii=False, indent=6)[:300]}")
        print()
