"""
Test Plan — MAVEN-ERE Seed Module
Module under test: embedding/seed_maven_ere.py
No network access needed — all tests use mock documents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import uuid
import pytest
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from unittest.mock import MagicMock
from causal_transform import CausalRelationRecord
from seed_maven_ere import parse_document, embed_and_store, RELATION_TYPES

# ---------------------------------------------------------------------------
# Mock MAVEN-ERE document factory (official THU-KEG dict format)
# ---------------------------------------------------------------------------

def make_doc(causal_relations=None, extra_events=None):
    """
    Minimal valid MAVEN-ERE document matching the official THU-KEG format.
    causal_relations is a dict: {"CAUSE": [[eid1, eid2], ...], ...}
    """
    doc = {
        "id": "doc_001",
        "title": "Weather Events",
        "sentences": [
            "The match was postponed because of a thunderstorm.",
            "The flooding caused widespread damage.",
        ],
        "events": [
            {"id": "E1", "type": "Catastrophe", "mention": [
                {"id": "M1", "sent_id": 0, "trigger_word": "thunderstorm", "offset": [0, 1]}
            ]},
            {"id": "E2", "type": "Change", "mention": [
                {"id": "M2", "sent_id": 0, "trigger_word": "postponed", "offset": [2, 3]}
            ]},
        ],
        "causal_relations": {
            "CAUSE": [["E1", "E2"]]
        } if causal_relations is None else causal_relations,
        "temporal_relations": {},
        "subevent_relations": {},
    }
    if extra_events:
        doc["events"].extend(extra_events)
    return doc


# ---------------------------------------------------------------------------
# Unit: parse_document — happy paths
# ---------------------------------------------------------------------------

class TestParseDocumentHappyPath:
    def test_returns_list(self):
        assert isinstance(parse_document(make_doc()), list)

    def test_one_cause_relation_yields_one_record(self):
        records = parse_document(make_doc())
        assert len(records) == 1

    def test_record_is_causal_relation_record(self):
        record = parse_document(make_doc())[0]
        assert isinstance(record, CausalRelationRecord)

    def test_policy_contains_cause_and_effect_triggers(self):
        record = parse_document(make_doc())[0]
        assert "thunderstorm" in record.policy
        assert "postponed" in record.policy

    def test_policy_relation_type_is_cause(self):
        record = parse_document(make_doc())[0]
        assert record.policy.startswith("CAUSE:")

    def test_embed_text_is_cause_sentence(self):
        record = parse_document(make_doc())[0]
        assert record.embed_text == make_doc()["sentences"][0]

    def test_bounded_context_is_doc_title(self):
        record = parse_document(make_doc())[0]
        assert record.bounded_context == "Weather Events"

    def test_aggregate_is_event_type(self):
        record = parse_document(make_doc())[0]
        assert record.aggregate == "Catastrophe"

    def test_id_contains_doc_id(self):
        record = parse_document(make_doc())[0]
        assert "doc_001" in record.id

    def test_precondition_relation_also_parsed(self):
        doc = make_doc(causal_relations={"PRECONDITION": [["E1", "E2"]]})
        records = parse_document(doc)
        assert len(records) == 1
        assert "PRECONDITION" in records[0].policy


# ---------------------------------------------------------------------------
# Unit: parse_document — boundary & failure modes
# ---------------------------------------------------------------------------

class TestParseDocumentBoundary:
    def test_no_causal_relations_returns_empty(self):
        doc = make_doc(causal_relations={})
        assert parse_document(doc) == []

    def test_temporal_relation_skipped(self):
        doc = make_doc(causal_relations={"BEFORE": [["E1", "E2"]]})
        assert parse_document(doc) == []

    def test_missing_head_id_skipped(self):
        doc = make_doc(causal_relations={"CAUSE": [["MISSING", "E2"]]})
        assert parse_document(doc) == []

    def test_missing_tail_id_skipped(self):
        doc = make_doc(causal_relations={"CAUSE": [["E1", "MISSING"]]})
        assert parse_document(doc) == []

    def test_tokenized_sentence_joined(self):
        doc = make_doc()
        doc["sentences"][0] = ["The", "match", "was", "postponed"]
        record = parse_document(doc)[0]
        assert record.embed_text == "The match was postponed"

    def test_empty_document_returns_empty(self):
        doc = {"id": "x", "title": "x", "sentences": [], "events": [],
               "causal_relations": {}, "temporal_relations": {}, "subevent_relations": {}}
        assert parse_document(doc) == []

    def test_multiple_relations_yields_multiple_records(self):
        doc = make_doc(causal_relations={
            "CAUSE": [["E1", "E2"]],
            "PRECONDITION": [["E2", "E1"]],
        })
        assert len(parse_document(doc)) == 2

    def test_record_ids_are_unique(self):
        doc = make_doc(causal_relations={
            "CAUSE": [["E1", "E2"], ["E2", "E1"]],
        })
        records = parse_document(doc)
        ids = [r.id for r in records]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Integration: embed_and_store -> Qdrant round-trip
# ---------------------------------------------------------------------------

EMBED_DIM = 384
COLLECTION_NAME = "test_maven"


def make_mock_model(seed=42):
    rng = np.random.default_rng(seed)
    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)
    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def _uuid(id_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


class TestEmbedAndStore:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self.model = make_mock_model()
        self.records = parse_document(make_doc())

        # Patch embed_and_store to use in-memory client
        import seed_maven_ere
        self._orig_qdrant_url = seed_maven_ere.QDRANT_URL
        # We'll monkey-patch QdrantClient inside embed_and_store
        self._client = self.client

    def _run_embed_and_store(self, records=None, col=COLLECTION_NAME):
        """Run embed_and_store but with our in-memory client."""
        import seed_maven_ere
        orig = seed_maven_ere.QdrantClient

        def fake_client(url=None):
            return self._client

        seed_maven_ere.QdrantClient = fake_client
        try:
            result = embed_and_store(records if records is not None else self.records,
                                     col, self.model)
        finally:
            seed_maven_ere.QdrantClient = orig
        return result

    def test_written_count_matches(self):
        written = self._run_embed_and_store()
        assert written == len(self.records)

    def test_empty_records_writes_zero(self):
        assert self._run_embed_and_store(records=[]) == 0

    def test_retrieve_by_id(self):
        self._run_embed_and_store()
        record = self.records[0]
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        assert len(results) == 1
        assert results[0].id == _uuid(record.id)

    def test_metadata_has_policy(self):
        self._run_embed_and_store()
        record = self.records[0]
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        meta = results[0].payload
        assert "policy" in meta
        assert "thunderstorm" in meta["policy"]

    def test_metadata_has_bounded_context(self):
        self._run_embed_and_store()
        record = self.records[0]
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        meta = results[0].payload
        assert meta["bounded_context"] == "Weather Events"

    def test_vector_dimension(self):
        self._run_embed_and_store()
        record = self.records[0]
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_uuid(record.id)],
            with_vectors=True,
        )
        assert len(results[0].vector) == EMBED_DIM

    def test_document_matches_embed_text(self):
        self._run_embed_and_store()
        record = self.records[0]
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        assert results[0].payload["document"] == record.embed_text
