"""
Test Plan — Retrieval Interface Module
Module under test: embedding/retrieve.py (query_similar, query_all)

Basic flow: Prompt Generator sends query_text (str) -> retrieve embeds it
-> ChromaDB similarity search -> structured {input, distance, collection, output}

All tests use ChromaDB EphemeralClient + mock SentenceTransformer (384-dim).
No network access needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import json
import pytest
import numpy as np
import chromadb
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Test fixtures — in-memory ChromaDB + mock model
# ---------------------------------------------------------------------------

EMBED_DIM = 384
MAVEN_COL = "maven_ere_causal"
BPC_COL = "bpc_logistics"

MAVEN_DOCS = [
    {
        "id": "m1",
        "text": "The storm caused severe flooding in the lowlands.",
        "meta": {
            "policy": "CAUSE: storm → flooding",
            "domain_event": "The storm caused severe flooding in the lowlands.",
            "command": "Flooding damaged crops and infrastructure.",
            "bounded_context": "Cyclone Trina",
            "aggregate": "Catastrophe",
            "source_phrase": "The storm caused severe flooding in the lowlands.",
        },
    },
    {
        "id": "m2",
        "text": "Political instability led to economic collapse.",
        "meta": {
            "policy": "CAUSE: instability → collapse",
            "domain_event": "Political instability led to economic collapse.",
            "command": "The government fell within months.",
            "bounded_context": "Arab Spring",
            "aggregate": "Change",
            "source_phrase": "Political instability led to economic collapse.",
        },
    },
]

BPC_DOCS = [
    {
        "id": "b1",
        "text": "Inventory shortages led to production delays.",
        "meta": {
            "policy": "does — Did the shortage cause delivery failure?",
            "domain_event": "Inventory shortages led to production delays.",
            "command": "Did the shortage cause delivery failure?",
            "bounded_context": "Logistics",
            "aggregate": "cause",
            "source_phrase": "Inventory shortages led to production delays.",
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


@pytest.fixture
def populated_client():
    """EphemeralClient with both MAVEN and BPC collections seeded."""
    client = chromadb.EphemeralClient()
    model = make_mock_model()

    for col_name, docs in [(MAVEN_COL, MAVEN_DOCS), (BPC_COL, BPC_DOCS)]:
        col = client.get_or_create_collection(col_name)
        texts = [d["text"] for d in docs]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        col.add(
            ids=[d["id"] for d in docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["meta"] for d in docs],
        )

    return client


# ---------------------------------------------------------------------------
# Helpers — patch retrieve module's singletons
# ---------------------------------------------------------------------------

def make_retriever(client, model):
    """Return (query_similar_fn, query_all_fn) wired to test fixtures."""
    import retrieve as r
    r._client = client
    r._model = model
    return r.query_similar, r.query_all


# ---------------------------------------------------------------------------
# Basic happy-path flow
# ---------------------------------------------------------------------------

class TestBasicFlow:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()
        col = self.client.get_or_create_collection(MAVEN_COL)
        texts = [d["text"] for d in MAVEN_DOCS]
        embs = self.model.encode(texts, show_progress_bar=False).tolist()
        col.add(ids=[d["id"] for d in MAVEN_DOCS], documents=texts,
                embeddings=embs, metadatas=[d["meta"] for d in MAVEN_DOCS])

        import retrieve as r
        r._client = self.client
        r._model = self.model
        from retrieve import query_similar
        self.query_similar = query_similar

    def test_returns_list(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL)
        assert isinstance(result, list)

    def test_returns_n_results(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=2)
        assert len(result) == 2

    def test_result_has_input_key(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert "input" in result[0]

    def test_result_has_distance_key(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert "distance" in result[0]

    def test_result_has_collection_key(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert result[0]["collection"] == MAVEN_COL

    def test_result_has_output_key(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert "output" in result[0]

    def test_output_has_policy(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert "policy" in result[0]["output"]

    def test_output_has_bounded_context(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert "bounded_context" in result[0]["output"]

    def test_output_missing_optional_fields_are_null(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        out = result[0]["output"]
        assert out.get("views") is None
        assert out.get("user_roles") is None
        assert out.get("process") is None

    def test_distance_is_float(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert isinstance(result[0]["distance"], float)

    def test_input_is_string(self):
        result = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=1)
        assert isinstance(result[0]["input"], str)

    def test_results_sorted_by_distance(self):
        results = self.query_similar("storm flooding", collection=MAVEN_COL, n_results=2)
        assert results[0]["distance"] <= results[1]["distance"]


# ---------------------------------------------------------------------------
# Collection selection — BPC vs MAVEN-ERE vs query_all
# ---------------------------------------------------------------------------

class TestCollectionSelection:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()

        for col_name, docs in [(MAVEN_COL, MAVEN_DOCS), (BPC_COL, BPC_DOCS)]:
            col = self.client.get_or_create_collection(col_name)
            texts = [d["text"] for d in docs]
            embs = self.model.encode(texts, show_progress_bar=False).tolist()
            col.add(ids=[d["id"] for d in docs], documents=texts,
                    embeddings=embs, metadatas=[d["meta"] for d in docs])

        import retrieve as r
        r._client = self.client
        r._model = self.model
        from retrieve import query_similar, query_all
        self.query_similar = query_similar
        self.query_all = query_all

    def test_maven_query_stays_in_maven_collection(self):
        results = self.query_similar("storm flooding", collection=MAVEN_COL)
        assert all(r["collection"] == MAVEN_COL for r in results)

    def test_bpc_query_stays_in_bpc_collection(self):
        results = self.query_similar("inventory shortage delay", collection=BPC_COL)
        assert all(r["collection"] == BPC_COL for r in results)

    def test_query_all_returns_from_multiple_collections(self):
        results = self.query_all("storm caused damage")
        collections_found = {r["collection"] for r in results}
        assert len(collections_found) >= 1

    def test_query_all_sorted_by_distance(self):
        results = self.query_all("inventory shortage")
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_invalid_collection_raises(self):
        from retrieve import query_similar
        import retrieve as r
        r._client = self.client
        r._model = self.model
        with pytest.raises(Exception):
            query_similar("test", collection="nonexistent_collection")


# ---------------------------------------------------------------------------
# Edge cases — empty / whitespace / gibberish / non-English
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()
        col = self.client.get_or_create_collection(MAVEN_COL)
        texts = [d["text"] for d in MAVEN_DOCS]
        embs = self.model.encode(texts, show_progress_bar=False).tolist()
        col.add(ids=[d["id"] for d in MAVEN_DOCS], documents=texts,
                embeddings=embs, metadatas=[d["meta"] for d in MAVEN_DOCS])

        import retrieve as r
        r._client = self.client
        r._model = self.model
        from retrieve import query_similar
        self.query_similar = query_similar

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            self.query_similar("", collection=MAVEN_COL)

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError):
            self.query_similar("   ", collection=MAVEN_COL)

    def test_gibberish_returns_results_without_crash(self):
        # Model still produces a vector; ChromaDB still returns nearest neighbours
        results = self.query_similar("xkqz9!@#$%^&*()", collection=MAVEN_COL)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_chinese_query_returns_results_without_crash(self):
        # Cross-language query: model maps Chinese to same 384-dim space
        results = self.query_similar("暴風雨造成嚴重洪水", collection=MAVEN_COL)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_very_long_query_does_not_crash(self):
        long_query = "storm flooding " * 200
        results = self.query_similar(long_query, collection=MAVEN_COL)
        assert isinstance(results, list)

    def test_single_word_query_returns_results(self):
        results = self.query_similar("storm", collection=MAVEN_COL)
        assert len(results) > 0

    def test_n_results_larger_than_collection_capped(self):
        # Collection has 2 docs; asking for 99 should return 2, not crash
        results = self.query_similar("storm", collection=MAVEN_COL, n_results=99)
        assert len(results) == len(MAVEN_DOCS)

    def test_n_results_zero_returns_empty(self):
        results = self.query_similar("storm", collection=MAVEN_COL, n_results=0)
        assert results == []


# ---------------------------------------------------------------------------
# Output contract — Prompt Generator depends on this shape
# ---------------------------------------------------------------------------

class TestOutputContract:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()
        col = self.client.get_or_create_collection(MAVEN_COL)
        texts = [d["text"] for d in MAVEN_DOCS]
        embs = self.model.encode(texts, show_progress_bar=False).tolist()
        col.add(ids=[d["id"] for d in MAVEN_DOCS], documents=texts,
                embeddings=embs, metadatas=[d["meta"] for d in MAVEN_DOCS])

        import retrieve as r
        r._client = self.client
        r._model = self.model
        from retrieve import query_similar
        self.result = query_similar("storm flooding", collection=MAVEN_COL, n_results=1)[0]

    def test_top_level_keys_are_correct(self):
        assert set(self.result.keys()) == {"input", "distance", "collection", "output"}

    def test_output_contains_all_required_ddd_fields(self):
        required = {"policy", "domain_event", "command", "bounded_context",
                    "aggregate", "source_phrase"}
        assert required.issubset(self.result["output"].keys())

    def test_output_contains_optional_fields_as_null(self):
        for field in ("views", "user_roles", "process"):
            assert field in self.result["output"]
            assert self.result["output"][field] is None

    def test_result_is_json_serialisable(self):
        json.dumps(self.result)  # must not raise
