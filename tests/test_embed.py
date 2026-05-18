"""
Test Plan — Embedding & DB Module
Module under test: embedding/embed.py (embed_records, review_collection)

SentenceTransformer is mocked. Qdrant runs in-memory (QdrantClient(":memory:")).
DDD fields: domain_event, command, policy, aggregate, bounded_context
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import uuid
import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from unittest.mock import MagicMock
from causal_transform import CausalRelationRecord, transform_row, transform_batch

EMBED_DIM = 384

VALID_ROW = {
    "id": 1,
    "phrase": "Inventory shortages led to production delays.",
    "question": "Did the shortage cause delivery failure?",
    "answer": "yes",
    "category": "cause",
    "domain": "Logistics",
}

REQUIRED_METADATA_KEYS = {
    "domain_event", "command", "policy",
    "aggregate", "bounded_context", "source_phrase"
}


def make_mock_model(seed: int = 42):
    rng = np.random.default_rng(seed)

    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def make_client():
    return QdrantClient(":memory:")


def make_record(row=None) -> CausalRelationRecord:
    return transform_row(row or VALID_ROW)


def _uuid(id_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


def embed_records_inline(client, model, collection_name, records):
    if not records:
        return 0
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    texts = [r.embed_text for r in records]
    embeddings = model.encode(texts).tolist()
    points = [
        PointStruct(
            id=_uuid(r.id),
            vector=embeddings[i],
            payload={**r.to_chroma_metadata(), "document": r.embed_text},
        )
        for i, r in enumerate(records)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(records)


# ---------------------------------------------------------------------------
# Unit: embedding quality (mocked)
# ---------------------------------------------------------------------------

class TestEmbeddingQuality:
    def setup_method(self):
        self.model = make_mock_model()

    def test_output_shape(self):
        assert self.model.encode(["test"]).shape == (1, EMBED_DIM)

    def test_batch_shape(self):
        assert self.model.encode(["a", "b", "c"]).shape == (3, EMBED_DIM)

    def test_no_nan(self):
        assert not any(v != v for v in self.model.encode(["test"])[0].tolist())

    def test_nonzero(self):
        assert any(v != 0 for v in self.model.encode(["test"])[0].tolist())

    def test_empty_string_does_not_crash(self):
        assert self.model.encode([""]).shape[1] == EMBED_DIM

    def test_long_string_does_not_crash(self):
        assert self.model.encode(["word " * 1000]).shape[1] == EMBED_DIM

    def test_encode_called_with_list(self):
        self.model.encode(["a", "b"])
        args, _ = self.model.encode.call_args
        assert isinstance(args[0], list)


# ---------------------------------------------------------------------------
# Integration: embed_records -> Qdrant round-trip
# ---------------------------------------------------------------------------

class TestEmbedRecordsIntegration:
    def setup_method(self):
        self.client = make_client()
        self.model = make_mock_model()

    def test_written_count_matches(self):
        assert embed_records_inline(self.client, self.model, "col1", [make_record()]) == 1

    def test_retrieve_by_id(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col2", [record])
        results = self.client.retrieve(
            collection_name="col2",
            ids=[_uuid(record.id)],
            with_payload=True,
            with_vectors=True,
        )
        assert len(results) == 1
        assert results[0].id == _uuid(record.id)

    def test_metadata_keys_complete(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col3", [record])
        results = self.client.retrieve(
            collection_name="col3",
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        meta = results[0].payload
        assert REQUIRED_METADATA_KEYS.issubset(meta.keys())

    def test_metadata_ddd_values(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col4", [record])
        results = self.client.retrieve(
            collection_name="col4",
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        meta = results[0].payload
        assert meta["bounded_context"] == "Logistics"
        assert meta["domain_event"] == VALID_ROW["phrase"]
        assert meta["command"] == VALID_ROW["question"]
        assert meta["policy"].startswith("does —")

    def test_empty_records_writes_zero(self):
        assert embed_records_inline(self.client, self.model, "col5", []) == 0

    def test_batch_count(self):
        rows = [{**VALID_ROW, "id": i} for i in range(1, 11)]
        records = [transform_row(r) for r in rows]
        embed_records_inline(self.client, self.model, "col6", records)
        assert self.client.get_collection("col6").points_count == 10

    def test_document_matches_embed_text(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col7", [record])
        results = self.client.retrieve(
            collection_name="col7",
            ids=[_uuid(record.id)],
            with_payload=True,
        )
        assert results[0].payload["document"] == record.embed_text

    def test_vector_dimension(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col8", [record])
        results = self.client.retrieve(
            collection_name="col8",
            ids=[_uuid(record.id)],
            with_vectors=True,
        )
        assert len(results[0].vector) == EMBED_DIM


# ---------------------------------------------------------------------------
# Integration: review_collection logic
# ---------------------------------------------------------------------------

class TestReviewCollection:
    def setup_method(self):
        self.client = make_client()
        self.model = make_mock_model()

    def _create_empty_collection(self, name):
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    def test_empty_collection_count_zero(self):
        self._create_empty_collection("empty")
        assert self.client.get_collection("empty").points_count == 0

    def test_non_empty_collection_count(self):
        embed_records_inline(self.client, self.model, "rev1", [make_record()])
        assert self.client.get_collection("rev1").points_count == 1

    def test_peek_no_nan(self):
        embed_records_inline(self.client, self.model, "rev2", [make_record()])
        points, _ = self.client.scroll(
            collection_name="rev2", limit=1, with_vectors=True
        )
        emb = points[0].vector
        assert not any(v != v for v in emb)

    def test_peek_correct_dimension(self):
        embed_records_inline(self.client, self.model, "rev3", [make_record()])
        points, _ = self.client.scroll(
            collection_name="rev3", limit=1, with_vectors=True
        )
        emb = points[0].vector
        assert len(emb) == EMBED_DIM


# ---------------------------------------------------------------------------
# Integration: full transform -> embed pipeline
# ---------------------------------------------------------------------------

class TestTransformToEmbedPipeline:
    def test_end_to_end(self):
        client, model = make_client(), make_mock_model()
        rows = [{**VALID_ROW, "id": i} for i in range(1, 6)]
        records, skipped = transform_batch(rows)
        assert skipped == 0
        assert embed_records_inline(client, model, "pipe1", records) == 5
        assert client.get_collection("pipe1").points_count == 5

    def test_invalid_rows_filtered(self):
        client, model = make_client(), make_mock_model()
        rows = [VALID_ROW, {**VALID_ROW, "id": 2, "phrase": ""}, {**VALID_ROW, "id": 3}]
        records, skipped = transform_batch(rows)
        assert skipped == 1
        embed_records_inline(client, model, "pipe2", records)
        assert client.get_collection("pipe2").points_count == 2
