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
                 "aggregate", "source_phrase", "trigger_span", "views", "user_roles", "process",
                 "commands_events_pairs"]


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


def _query_raw(query_text: str, collection: str, n_results: int) -> list[dict]:
    """Internal: query by exact full collection name, no prefix handling."""
    embedding = _get_model().encode([query_text], show_progress_bar=False).tolist()[0]
    client = _get_client()

    col_size = client.get_collection(collection).points_count
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


def query_similar(
    query_text: str,
    collection: str = "maven_ere_causal",
    n_results: int = 3,
) -> list[dict]:
    """
    Embed query_text and return n_results most similar records from one collection.
    COLLECTION_PREFIX env var is automatically prepended to collection.
    Raises ValueError for empty/whitespace-only query.
    Raises exception if collection does not exist.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    prefix = os.getenv("COLLECTION_PREFIX", "")
    return _query_raw(query_text, f"{prefix}{collection}", n_results)


def query_all(
    query_text: str,
    n_results: int = 3,
) -> list[dict]:
    """
    Search all available collections and return the top n_results most similar
    results across all collections.
    Useful when the Prompt Generator does not know which collection to target.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")

    all_results = []
    for col_name in list_collections():
        try:
            all_results.extend(
                _query_raw(query_text, col_name, n_results)
            )
        except Exception:
            continue

    all_results.sort(key=lambda r: r["distance"])
    return all_results[:n_results]


def query_multiple(
    query_text: str,
    collections: list[str],
    n_results: int = 3,
) -> list[dict]:
    """
    Search a specific list of collections and return the top n_results most similar
    results across all specified collections.
    COLLECTION_PREFIX env var is automatically prepended to each collection name.

    Example:
        query_multiple(
            "User places an order",
            collections=["mtop_commands", "log_commands"],
            n_results=3,
        )
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")
    if not collections:
        raise ValueError("collections must not be empty")

    prefix = os.getenv("COLLECTION_PREFIX", "")
    all_results = []
    for col_name in collections:
        try:
            all_results.extend(
                _query_raw(query_text, f"{prefix}{col_name}", n_results)
            )
        except Exception:
            continue

    all_results.sort(key=lambda r: r["distance"])
    return all_results[:n_results]


def query_by_prefix(
    query_text: str,
    prefixes: list[str],
    n_results_per_collection: int = 3,
) -> list[dict]:
    """
    Search all collections whose names match any of the given prefixes.
    COLLECTION_PREFIX env var is automatically prepended to each prefix.

    Example:
        # Search all bpc_* collections
        query_by_prefix("User places an order", prefixes=["bpc"])

        # Search both bpc_* and maven_ere_* collections
        query_by_prefix("User places an order", prefixes=["bpc", "maven_ere"])
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must not be empty")
    if not prefixes:
        raise ValueError("prefixes must not be empty")

    collection_prefix = os.getenv("COLLECTION_PREFIX", "")
    full_prefixes = [f"{collection_prefix}{p}" for p in prefixes]

    matched = [
        col for col in list_collections()
        if any(col.startswith(fp) for fp in full_prefixes)
    ]

    all_results = []
    for col_name in matched:
        try:
            all_results.extend(
                _query_raw(query_text, col_name, n_results_per_collection)
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
    import datetime
    import pathlib

    prefix = os.getenv("COLLECTION_PREFIX", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = pathlib.Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    md_path  = reports_dir / f"retrieve_demo_{timestamp}.md"
    json_path = reports_dir / f"retrieve_demo_{timestamp}.json"

    _md_lines: list[str] = []
    _all_results: dict = {}

    def _w(line: str = ""):
        """Append a line to the markdown report buffer."""
        _md_lines.append(line)

    def _section(title: str):
        print(f"  >> {title}")
        _w(f"\n## {title}")

    def _run_query(label: str, query_fn, *args, **kwargs) -> list[dict]:
        results = query_fn(*args, **kwargs)
        _all_results[label] = results

        _w(f"\n**Query:** {kwargs.get('query_text') or args[0]}\n")
        _w("| Rank | Distance | Collection | Input (truncated) | domain_event | command | policy | bounded_context |")
        _w("|------|----------|------------|-------------------|--------------|---------|--------|-----------------|")
        for rank, r in enumerate(results, 1):
            o = r["output"]
            inp = r["input"].replace("|", "/").replace("\n", " ")[:80]
            de  = (o["domain_event"] or "").replace("|", "/")
            cmd = (o["command"]      or "").replace("|", "/")
            pol = (o["policy"]       or "").replace("|", "/")[:60]
            bc  = (o["bounded_context"] or "").replace("|", "/")
            _w(f"| {rank} | {r['distance']} | {r['collection']} | {inp} | {de} | {cmd} | {pol} | {bc} |")
        return results

    print("=== Retrieval Demo ===")
    all_cols = list_collections()
    our_cols = set(c for c in all_cols if c.startswith(prefix)) if prefix else set(all_cols)
    print(f"Collections found : {len(our_cols)}  {sorted(our_cols)}")
    print(f"Writing report to : {md_path.name}")
    print(f"Writing JSON to   : {json_path.name}\n")

    if not our_cols:
        print("No collections found. Run embed.py, seed_maven_ere.py, seed_mtop.py, seed_log.py first.")
        raise SystemExit(1)

    _w(f"# Retrieval Demo — {timestamp}")
    _w(f"\nPrefix: `{prefix or '(none)'}` | Collections: {sorted(our_cols)}\n")

    # --- DATASET 1: MAVEN-ERE ---
    if f"{prefix}maven_ere_causal" in our_cols:
        _section("DATASET 1: maven_ere_causal")
        _run_query("maven_ere_causal", query_by_prefix,
                   "The storm caused severe flooding in the region.",
                   prefixes=["maven_ere"], n_results_per_collection=5)
    else:
        print("  [SKIP] maven_ere_causal")
        _w("\n## DATASET 1: maven_ere_causal — SKIPPED (run seed_maven_ere.py)")

    # --- DATASET 2: BPC ---
    bpc_cols = sorted(c for c in our_cols if c.startswith(f"{prefix}bpc_"))
    if bpc_cols:
        _section("DATASET 2: BPC (all domains)")
        _run_query("bpc_sample", query_by_prefix,
                   "Inventory shortages led to production delays.",
                   prefixes=["bpc"], n_results_per_collection=2)
        _w(f"\n> All BPC collections: {bpc_cols}")
    else:
        print("  [SKIP] bpc_*")
        _w("\n## DATASET 2: BPC — SKIPPED (run embed.py)")

    # --- DATASET 3: MTOP ---
    if f"{prefix}mtop_commands" in our_cols:
        _section("DATASET 3: mtop_commands")
        _run_query("mtop_commands", query_by_prefix,
                   "Schedule a meeting with the team for tomorrow morning.",
                   prefixes=["mtop"], n_results_per_collection=5)
    else:
        print("  [SKIP] mtop_commands")
        _w("\n## DATASET 3: mtop_commands — SKIPPED (run seed_mtop.py)")

    # --- DATASET 4a: LOG domain events ---
    if f"{prefix}log_domain_events" in our_cols:
        _section("DATASET 4a: log_domain_events")
        _run_query("log_domain_events", query_by_prefix,
                   "The user logs into the system and is authenticated.",
                   prefixes=["log_domain_events"], n_results_per_collection=5)
    else:
        print("  [SKIP] log_domain_events")
        _w("\n## DATASET 4a: log_domain_events — SKIPPED (run seed_log.py)")

    # --- DATASET 4b: LOG commands ---
    if f"{prefix}log_commands" in our_cols:
        _section("DATASET 4b: log_commands")
        _run_query("log_commands", query_by_prefix,
                   "The user starts a meeting and invites participants.",
                   prefixes=["log_commands"], n_results_per_collection=5)
    else:
        print("  [SKIP] log_commands")
        _w("\n## DATASET 4b: log_commands — SKIPPED (run seed_log.py)")

    # --- DATASET 4c: LOG command-event pairs ---
    if f"{prefix}log_commands_events_pairs" in our_cols:
        _section("DATASET 4c: log_commands_events_pairs")
        _run_query("log_commands_events_pairs", query_by_prefix,
                   "The user initiates a meeting and it gets scheduled.",
                   prefixes=["log_commands_events_pairs"], n_results_per_collection=5)
    else:
        print("  [SKIP] log_commands_events_pairs")
        _w("\n## DATASET 4c: log_commands_events_pairs — SKIPPED (run seed_log.py)")

    # --- CROSS-COLLECTION: query_all ---
    _section("CROSS-COLLECTION: query_all")
    _run_query("query_all", query_all,
               "A user initiates an action and the system responds with a status change.",
               n_results=2)

    # --- query_by_prefix: log only ---
    _section("query_by_prefix: prefix=['log']")
    _run_query("query_by_prefix_log", query_by_prefix,
               "The meeting is cancelled by the initiator.",
               prefixes=["log"], n_results_per_collection=3)

    # --- query_multiple: mtop + log_commands ---
    if f"{prefix}mtop_commands" in our_cols and f"{prefix}log_commands" in our_cols:
        _section("query_multiple: mtop_commands + log_commands")
        _run_query("query_multiple", query_multiple,
                   "User cancels a scheduled meeting.",
                   collections=["mtop_commands", "log_commands"],
                   n_results=3)

    # --- Write files ---
    md_path.write_text("\n".join(_md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(_all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Report : {md_path}")
    print(f"  JSON   : {json_path}")
