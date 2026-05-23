"""
Test Plan — Retrieval against log_domain_events and log_commands collections
Tests cover query_similar, query_multiple, and query_by_prefix for the two
new DDD log collections.

Key payload differences from other collections:
  log_domain_events : domain_event filled, command null
  log_commands      : command filled, user_roles filled (when actor known), domain_event null
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import uuid
import json
import pytest
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixtures — realistic log data
# ---------------------------------------------------------------------------

EMBED_DIM = 384
LOG_EVENTS_COL = "log_domain_events"
LOG_COMMANDS_COL = "log_commands"

LOG_EVENT_DOCS = [
    {
        "id": "e1",
        "text": "The user is logged in to the system .",
        "meta": {
            "domain_event": "user logged in",
            "source_phrase": "The user is logged in to the system .",
        },
    },
    {
        "id": "e2",
        "text": "The system verifies the user .",
        "meta": {
            "domain_event": "user verified",
            "source_phrase": "The system verifies the user .",
        },
    },
    {
        "id": "e3",
        "text": "The initiated meeting is scheduled and the system sends new meeting messages to all participants .",
        "meta": {
            "domain_event": "meeting scheduled",
            "source_phrase": "The initiated meeting is scheduled and the system sends new meeting messages to all participants .",
        },
    },
    {
        "id": "e4",
        "text": "The system sends a request to the backend to cancel the meeting .",
        "meta": {
            "domain_event": "meeting canceled",
            "source_phrase": "The system sends a request to the backend to cancel the meeting .",
        },
    },
]

LOG_COMMAND_DOCS = [
    {
        "id": "c1",
        "text": "The User enters the user's username and password and presses the LOGIN button .",
        "meta": {
            "command": "login",
            "user_roles": "user",
            "source_phrase": "The User enters the user's username and password and presses the LOGIN button .",
        },
    },
    {
        "id": "c2",
        "text": "The user clicks INITIATE_MEETING button .",
        "meta": {
            "command": "initiate meeting",
            "user_roles": "initiator",
            "source_phrase": "The user clicks INITIATE_MEETING button .",
        },
    },
    {
        "id": "c3",
        "text": "The user clicks CANCEL_MEETING .",
        "meta": {
            "command": "cancel meeting",
            "user_roles": "initiator",
            "source_phrase": "The user clicks CANCEL_MEETING .",
        },
    },
    {
        "id": "c4",
        "text": "User is registered in the system and has the username and password to access the system .",
        "meta": {
            "command": "register user",
            "source_phrase": "User is registered in the system and has the username and password to access the system .",
        },
    },
]


def make_mock_model(seed: int = 42):
    rng = np.random.default_rng(seed)

    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def _uuid(id_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


def _seed_collection(client, col_name, docs, model):
    existing = [c.name for c in client.get_collections().collections]
    if col_name not in existing:
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    points = [
        PointStruct(
            id=_uuid(d["id"]),
            vector=embeddings[i],
            payload={**d["meta"], "document": d["text"]},
        )
        for i, d in enumerate(docs)
    ]
    client.upsert(collection_name=col_name, points=points)


def _wire(client, model):
    import retrieve as r
    r._client = client
    r._model = model


# ---------------------------------------------------------------------------
# query_similar — log_domain_events
# ---------------------------------------------------------------------------

class TestQuerySimilarDomainEvents:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.model = make_mock_model()
        _seed_collection(self.client, LOG_EVENTS_COL, LOG_EVENT_DOCS, self.model)
        _wire(self.client, self.model)
        from retrieve import query_similar
        self.query_similar = query_similar

    def test_returns_results(self):
        results = self.query_similar("user logs into the system", collection=LOG_EVENTS_COL)
        assert len(results) > 0

    def test_collection_label_is_log_events(self):
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        assert results[0]["collection"] == LOG_EVENTS_COL

    def test_domain_event_field_is_populated(self):
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        assert results[0]["output"]["domain_event"] is not None
        assert results[0]["output"]["domain_event"] != ""

    def test_command_field_is_null(self):
        # log_domain_events records have no command
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        assert results[0]["output"]["command"] is None

    def test_user_roles_is_null(self):
        results = self.query_similar("meeting was scheduled", collection=LOG_EVENTS_COL, n_results=1)
        assert results[0]["output"]["user_roles"] is None

    def test_policy_is_null(self):
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        assert results[0]["output"]["policy"] is None

    def test_output_shape_complete(self):
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        from retrieve import OUTPUT_FIELDS
        assert set(OUTPUT_FIELDS).issubset(results[0]["output"].keys())

    def test_result_is_json_serialisable(self):
        results = self.query_similar("user logs in", collection=LOG_EVENTS_COL, n_results=1)
        json.dumps(results[0])


# ---------------------------------------------------------------------------
# query_similar — log_commands
# ---------------------------------------------------------------------------

class TestQuerySimilarCommands:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.model = make_mock_model()
        _seed_collection(self.client, LOG_COMMANDS_COL, LOG_COMMAND_DOCS, self.model)
        _wire(self.client, self.model)
        from retrieve import query_similar
        self.query_similar = query_similar

    def test_returns_results(self):
        results = self.query_similar("user initiates a meeting", collection=LOG_COMMANDS_COL)
        assert len(results) > 0

    def test_collection_label_is_log_commands(self):
        results = self.query_similar("user logs in", collection=LOG_COMMANDS_COL, n_results=1)
        assert results[0]["collection"] == LOG_COMMANDS_COL

    def test_command_field_is_populated(self):
        results = self.query_similar("user initiates meeting", collection=LOG_COMMANDS_COL, n_results=1)
        assert results[0]["output"]["command"] is not None
        assert results[0]["output"]["command"] != ""

    def test_domain_event_field_is_null(self):
        # log_commands records have no domain_event
        results = self.query_similar("user initiates meeting", collection=LOG_COMMANDS_COL, n_results=1)
        assert results[0]["output"]["domain_event"] is None

    def test_user_roles_populated_when_actor_known(self):
        # login, initiate meeting, cancel meeting all have actors
        all_results = self.query_similar("user clicks button", collection=LOG_COMMANDS_COL, n_results=3)
        roles = [r["output"]["user_roles"] for r in all_results]
        assert any(role is not None for role in roles)

    def test_user_roles_null_when_actor_unknown(self):
        # "register user" has no actor — seed it alone and verify
        client = QdrantClient(":memory:")
        model = make_mock_model(seed=99)
        _seed_collection(client, LOG_COMMANDS_COL, [LOG_COMMAND_DOCS[3]], model)
        _wire(client, model)
        from retrieve import query_similar
        results = query_similar("register user", collection=LOG_COMMANDS_COL, n_results=1)
        assert results[0]["output"]["user_roles"] is None


# ---------------------------------------------------------------------------
# query_multiple — both log collections together
# ---------------------------------------------------------------------------

class TestQueryMultiple:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.model = make_mock_model()
        _seed_collection(self.client, LOG_EVENTS_COL, LOG_EVENT_DOCS, self.model)
        _seed_collection(self.client, LOG_COMMANDS_COL, LOG_COMMAND_DOCS, self.model)
        _wire(self.client, self.model)
        from retrieve import query_multiple
        self.query_multiple = query_multiple

    def test_returns_results_from_both_collections(self):
        results = self.query_multiple(
            "user logs into the system",
            collections=[LOG_EVENTS_COL, LOG_COMMANDS_COL],
        )
        collections_hit = {r["collection"] for r in results}
        assert LOG_EVENTS_COL in collections_hit
        assert LOG_COMMANDS_COL in collections_hit

    def test_results_sorted_by_distance(self):
        results = self.query_multiple(
            "meeting is scheduled",
            collections=[LOG_EVENTS_COL, LOG_COMMANDS_COL],
        )
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_empty_collections_list_raises(self):
        from retrieve import query_multiple
        import retrieve as r
        r._client = self.client
        r._model = self.model
        with pytest.raises(ValueError):
            query_multiple("meeting", collections=[])

    def test_mixed_payload_shapes_in_same_result_set(self):
        results = self.query_multiple(
            "meeting scheduled by initiator",
            collections=[LOG_EVENTS_COL, LOG_COMMANDS_COL],
            n_results_per_collection=2,
        )
        event_results = [r for r in results if r["collection"] == LOG_EVENTS_COL]
        cmd_results = [r for r in results if r["collection"] == LOG_COMMANDS_COL]

        # event records have domain_event, no command
        for r in event_results:
            assert r["output"]["domain_event"] is not None
            assert r["output"]["command"] is None

        # command records have command, no domain_event
        for r in cmd_results:
            assert r["output"]["command"] is not None
            assert r["output"]["domain_event"] is None


# ---------------------------------------------------------------------------
# query_by_prefix — "log" prefix covers both collections
# ---------------------------------------------------------------------------

class TestQueryByPrefix:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.model = make_mock_model()
        _seed_collection(self.client, LOG_EVENTS_COL, LOG_EVENT_DOCS, self.model)
        _seed_collection(self.client, LOG_COMMANDS_COL, LOG_COMMAND_DOCS, self.model)
        _wire(self.client, self.model)
        from retrieve import query_by_prefix
        self.query_by_prefix = query_by_prefix

    def test_prefix_log_hits_both_collections(self):
        results = self.query_by_prefix("user logs in", prefixes=["log"])
        collections_hit = {r["collection"] for r in results}
        assert LOG_EVENTS_COL in collections_hit
        assert LOG_COMMANDS_COL in collections_hit

    def test_prefix_log_domain_events_only(self):
        results = self.query_by_prefix("user verified", prefixes=["log_domain"])
        assert all(r["collection"] == LOG_EVENTS_COL for r in results)

    def test_prefix_log_commands_only(self):
        results = self.query_by_prefix("cancel meeting", prefixes=["log_commands"])
        assert all(r["collection"] == LOG_COMMANDS_COL for r in results)

    def test_results_sorted_by_distance(self):
        results = self.query_by_prefix("user action", prefixes=["log"])
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_nonmatching_prefix_returns_empty(self):
        results = self.query_by_prefix("user action", prefixes=["bpc_", "maven"])
        assert results == []

    def test_empty_prefixes_raises(self):
        from retrieve import query_by_prefix
        import retrieve as r
        r._client = self.client
        r._model = self.model
        with pytest.raises(ValueError):
            query_by_prefix("meeting", prefixes=[])


# ---------------------------------------------------------------------------
# Demo — run directly against real local Qdrant to show terminal output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("LOG DATASET RETRIEVAL DEMO (real local Qdrant)")
    print("=" * 60)
    print("Requires: Qdrant running + seed_log.py already executed\n")

    import retrieve as r
    r._client = None
    r._model = None
    from retrieve import query_similar, query_multiple, query_by_prefix, list_collections

    cols = list_collections()
    log_cols = [c for c in cols if c.startswith("log_")]
    print(f"Collections on server : {cols}")
    print(f"Log collections found : {log_cols}\n")

    if not log_cols:
        print("No log collections found. Run: python embedding/seed_log.py")
        raise SystemExit(1)

    # --- DEMO 1: query log_domain_events ---
    if "log_domain_events" in log_cols:
        print("=" * 60)
        print("DEMO 1: query_similar -> log_domain_events")
        print("=" * 60)
        q = "The user logs into the system and is authenticated."
        print(f"Query: \"{q}\"\n")
        for rank, res in enumerate(query_similar(q, collection="log_domain_events", n_results=3), 1):
            print(f"  Rank {rank}  distance={res['distance']}")
            print(f"    input        : {res['input']}")
            print(f"    domain_event : {res['output']['domain_event']}")
            print(f"    command      : {res['output']['command']}")
            print(f"    user_roles   : {res['output']['user_roles']}")
        print()

    # --- DEMO 2: query log_commands ---
    if "log_commands" in log_cols:
        print("=" * 60)
        print("DEMO 2: query_similar -> log_commands")
        print("=" * 60)
        q = "The user starts a meeting and invites participants."
        print(f"Query: \"{q}\"\n")
        for rank, res in enumerate(query_similar(q, collection="log_commands", n_results=3), 1):
            print(f"  Rank {rank}  distance={res['distance']}")
            print(f"    input        : {res['input']}")
            print(f"    command      : {res['output']['command']}")
            print(f"    user_roles   : {res['output']['user_roles']}")
            print(f"    domain_event : {res['output']['domain_event']}")
        print()

    # --- DEMO 3: query_multiple across both log collections ---
    if len(log_cols) >= 2:
        print("=" * 60)
        print("DEMO 3: query_multiple -> log_domain_events + log_commands")
        print("=" * 60)
        q = "The meeting is initiated and then cancelled."
        print(f"Query: \"{q}\"\n")
        for rank, res in enumerate(
            query_multiple(q, collections=["log_domain_events", "log_commands"], n_results_per_collection=2), 1
        ):
            print(f"  Rank {rank}  distance={res['distance']}  collection={res['collection']}")
            print(f"    input        : {res['input'][:90]}")
            print(f"    domain_event : {res['output']['domain_event']}")
            print(f"    command      : {res['output']['command']}")
            print(f"    user_roles   : {res['output']['user_roles']}")
        print()

    # --- DEMO 4: query_by_prefix ---
    print("=" * 60)
    print("DEMO 4: query_by_prefix(prefixes=['log'])")
    print("=" * 60)
    q = "The administrator configures the system settings."
    print(f"Query: \"{q}\"\n")
    for rank, res in enumerate(query_by_prefix(q, prefixes=["log"], n_results_per_collection=2), 1):
        print(f"  Rank {rank}  distance={res['distance']}  collection={res['collection']}")
        print(f"    input        : {res['input'][:90]}")
        print(f"    domain_event : {res['output']['domain_event']}")
        print(f"    command      : {res['output']['command']}")
        print(f"    user_roles   : {res['output']['user_roles']}")
    print()

    # --- DEMO 5: full JSON shape ---
    print("=" * 60)
    print("DEMO 5: full JSON output shape")
    print("=" * 60)
    res = query_similar(
        "user cancels a scheduled meeting",
        collection="log_domain_events" if "log_domain_events" in log_cols else log_cols[0],
        n_results=1,
    )[0]
    print(json.dumps(res, ensure_ascii=False, indent=2))
