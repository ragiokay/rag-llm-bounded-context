# Retrieval interface for Prompt Generator
# Flow: query_text (str) -> embed -> ChromaDB similarity search -> metadata list
#
# Prompt Generator usage:
#   from retrieve import query_similar
#   results = query_similar("The storm caused flooding", collection="maven_ere_causal")

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")

_model = None
_client = None


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


def query_similar(
    query_text: str,
    collection: str = "maven_ere_causal",
    n_results: int = 3,
) -> list[dict]:
    """
    Embed query_text and return the n_results most similar records.

    Returns a list of dicts, each containing:
        distance        — similarity score (lower = more similar)
        document        — the original embedded sentence (the "In")
        domain_event    — cause sentence
        command         — effect sentence
        policy          — e.g. "CAUSE: drought → famine"
        bounded_context — document/domain name
        aggregate       — event type or category
        source_phrase   — original source text
    """
    model = _get_model()
    client = _get_client()

    embedding = model.encode([query_text], show_progress_bar=False).tolist()

    col = client.get_collection(name=collection)
    results = col.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        entry = {
            "distance": round(results["distances"][0][i], 4),
            "document": results["documents"][0][i],
            **results["metadatas"][0][i],
        }
        output.append(entry)
    return output


def format_result(result: dict) -> str:
    """Pretty-print a single query result."""
    lines = [
        f"  Distance : {result['distance']}",
        f"  In       : {result['document'][:120]}",
        f"  Policy   : {result['policy']}",
        f"  Event    : {result['domain_event'][:80]}",
        f"  Command  : {result['command'][:80]}",
        f"  Context  : {result['bounded_context']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo — run directly to see live input/output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Retrieval Demo ===")
    print(f"Available collections: {list_collections()}\n")

    test_queries = [
        ("maven_ere_causal", "The storm caused severe flooding in the region."),
        ("maven_ere_causal", "Political instability led to economic collapse."),
        ("maven_ere_causal", "The army advanced and captured the territory."),
    ]

    for collection, query in test_queries:
        print(f"Query   : \"{query}\"")
        print(f"Collection: {collection}")
        results = query_similar(query, collection=collection, n_results=3)
        for rank, r in enumerate(results, 1):
            print(f"  --- Rank {rank} ---")
            print(format_result(r))
        print()
