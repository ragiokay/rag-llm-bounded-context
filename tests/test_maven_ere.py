"""
Test Plan — MAVEN-ERE Seed Module
Module under test: embedding/seed_maven_ere.py
No network access needed — all tests use mock documents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import pytest
import chromadb
import numpy as np
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
            {"id": "E1", "type": "Catastrophe", "mentions": [
                {"id": "M1", "sent_id": 0, "trigger_word": "thunderstorm", "offset": [0, 1]}
            ]},
            {"id": "E2", "type": "Change", "mentions": [
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
# Integration: embed_and_store -> ChromaDB round-trip
# ---------------------------------------------------------------------------

EMBED_DIM = 384


def make_mock_model(seed=42):
    rng = np.random.default_rng(seed)
    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)
    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


class TestEmbedAndStore:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()
        self.col = self.client.get_or_create_collection("test_maven")
        self.records = parse_document(make_doc())

    def test_written_count_matches(self):
        written = embed_and_store(self.records, self.col, self.model)
        assert written == len(self.records)

    def test_empty_records_writes_zero(self):
        assert embed_and_store([], self.col, self.model) == 0

    def test_retrieve_by_id(self):
        embed_and_store(self.records, self.col, self.model)
        result = self.col.get(ids=[self.records[0].id])
        assert result["ids"] == [self.records[0].id]

    def test_metadata_has_policy(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert "policy" in meta
        assert "thunderstorm" in meta["policy"]

    def test_metadata_has_bounded_context(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert meta["bounded_context"] == "Weather Events"

    def test_vector_dimension(self):
        embed_and_store(self.records, self.col, self.model)
        embs = self.col.get(
            ids=[self.records[0].id], include=["embeddings"]
        )["embeddings"]
        assert len(embs[0]) == EMBED_DIM

    def test_document_matches_embed_text(self):
        embed_and_store(self.records, self.col, self.model)
        docs = self.col.get(
            ids=[self.records[0].id], include=["documents"]
        )["documents"]
        assert docs[0] == self.records[0].embed_text
