# Retrieval interface for Prompt Generator
# Flow: query_text (str) -> embed -> Qdrant similarity search -> structured result
#
# Prompt Generator usage:
#   from retrieve import query_similar, query_all
#   results = query_similar("The storm caused flooding", collection="maven_ere_causal")
#   results = query_all("The storm caused flooding")  # search all collections

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_model = None
_client = None

OUTPUT_FIELDS = ["policy", "domain_event", "command", "bounded_context",
                 "aggregate", "source_phrase", "trigger_span", "views", "user_roles", "process"]


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=300)
    return _client


def list_collections() -> list[str]:
    """Return all collection names available in Qdrant."""
    return [c.name for c in _get_client().get_collections().collections]


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
    Raises exception if collection does not exist.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    embedding = _get_model().encode([query_text], show_progress_bar=False).tolist()[0]
    client = _get_client()

    col_info = client.get_collection(collection)
    col_size = col_info.points_count

    safe_n = min(n_results, col_size)
    if safe_n == 0:
        return []

    response = client.query_points(
        collection_name=collection,
        query=embedding,
        limit=safe_n,
        with_payload=True,
    )

    return [
        _build_result(
            round(1 - result.score, 4),
            result.payload.get("document", ""),
            result.payload,
            collection,
        )
        for result in response.points
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


def query_multiple(
    query_text: str,
    collections: list[str],
    n_results_per_collection: int = 3,
) -> list[dict]:
    """
    Search a specific list of collections and return merged results sorted by distance.
    Use this when you want to mix results from selected collections only.

    Example:
        query_multiple(
            "User places an order",
            collections=[
                "spring2026SE_g1_rag_mtop_commands",
                "spring2026SE_g1_rag_maven_ere_causal",
            ],
            n_results_per_collection=3,
        )
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")
    if not collections:
        raise ValueError("collections must not be empty")

    all_results = []
    for col_name in collections:
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

    prefix = os.getenv("COLLECTION_PREFIX", "")

    print("=== Retrieval Demo ===")
    all_cols = list_collections()
    cols = [c for c in all_cols if c.startswith(prefix)] if prefix else all_cols
    print(f"All collections on server : {len(all_cols)}")
    print(f"Our collections (prefix='{prefix}'): {cols}\n")

    if not cols:
        print("No collections found. Run embed.py (and optionally seed_maven_ere.py / seed_mtop.py) first.")
        raise SystemExit(1)

    # --- TEST 1: query_similar against the first available collection ---
    print("=" * 60)
    print(f"TEST 1: query_similar — {cols[0]}")
    print("=" * 60)
    q = "The storm caused severe flooding in the region."
    print(f"Query: \"{q}\"\n")
    for rank, r in enumerate(query_similar(q, collection=cols[0], n_results=3), 1):
        print(f"  Rank {rank}  distance={r['distance']}")
        print(f"    input  : {r['input'][:100]}")
        print(f"    policy : {r['output']['policy']}")
        print(f"    context: {r['output']['bounded_context']}")
    print()

    # --- TEST 2: query_similar — BPC if present, else second available collection ---
    bpc_cols = [c for c in cols if c.startswith("bpc_")]
    test2_col = bpc_cols[0] if bpc_cols else (cols[1] if len(cols) > 1 else cols[0])
    print("=" * 60)
    print(f"TEST 2: query_similar — {test2_col}")
    print("=" * 60)
    q = "Inventory shortages led to production delays."
    print(f"Query: \"{q}\"\n")
    for rank, r in enumerate(query_similar(q, collection=test2_col, n_results=3), 1):
        print(f"  Rank {rank}  distance={r['distance']}")
        print(f"    input  : {r['input'][:100]}")
        print(f"    policy : {r['output']['policy']}")
        print(f"    context: {r['output']['bounded_context']}")
    print()

    # --- TEST 3: query_all — cross-collection ---
    print("=" * 60)
    print("TEST 3: query_all — search all collections")
    print("=" * 60)
    q = "The drought caused crop failure and food shortages."
    print(f"Query: \"{q}\"\n")
    for rank, r in enumerate(query_all(q, n_results_per_collection=2), 1):
        print(f"  Rank {rank}  distance={r['distance']}  collection={r['collection']}")
        print(f"    input  : {r['input'][:100]}")
        print(f"    policy : {r['output']['policy']}")
    print()

    # --- TEST 4: full output shape (JSON) ---
    print("=" * 60)
    print("TEST 4: full output shape (JSON)")
    print("=" * 60)
    r = query_similar("Political crisis led to government collapse.",
                      collection=cols[0], n_results=1)[0]
    print(json.dumps(r, ensure_ascii=False, indent=2))
