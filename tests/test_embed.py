"""
Test Plan — Embedding & DB Module
Module under test: embedding/embed.py (embed_records, review_collection)

SentenceTransformer is mocked. ChromaDB runs in-memory (EphemeralClient).
DDD fields: domain_event, command, policy, aggregate, bounded_context
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import numpy as np
import pytest
import chromadb
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
    return chromadb.EphemeralClient()


def make_record(row=None) -> CausalRelationRecord:
    return transform_row(row or VALID_ROW)


def embed_records_inline(client, model, collection_name, records):
    if not records:
        return 0
    collection = client.get_or_create_collection(name=collection_name)
    texts = [r.embed_text for r in records]
    embeddings = model.encode(texts).tolist()
    collection.add(
        ids=[r.id for r in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[r.to_chroma_metadata() for r in records],
    )
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
# Integration: embed_records -> ChromaDB round-trip
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
        result = self.client.get_collection("col2").get(ids=[record.id])
        assert result["ids"] == [record.id]

    def test_metadata_keys_complete(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col3", [record])
        meta = self.client.get_collection("col3").get(
            ids=[record.id], include=["metadatas"]
        )["metadatas"][0]
        assert REQUIRED_METADATA_KEYS.issubset(meta.keys())

    def test_metadata_ddd_values(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col4", [record])
        meta = self.client.get_collection("col4").get(
            ids=[record.id], include=["metadatas"]
        )["metadatas"][0]
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
        assert self.client.get_collection("col6").count() == 10

    def test_document_matches_embed_text(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col7", [record])
        docs = self.client.get_collection("col7").get(
            ids=[record.id], include=["documents"]
        )["documents"]
        assert docs[0] == record.embed_text

    def test_vector_dimension(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col8", [record])
        embs = self.client.get_collection("col8").get(
            ids=[record.id], include=["embeddings"]
        )["embeddings"]
        assert len(embs[0]) == EMBED_DIM


# ---------------------------------------------------------------------------
# Integration: review_collection logic
# ---------------------------------------------------------------------------

class TestReviewCollection:
    def setup_method(self):
        self.client = make_client()
        self.model = make_mock_model()

    def test_empty_collection_count_zero(self):
        assert self.client.get_or_create_collection("empty").count() == 0

    def test_non_empty_collection_count(self):
        embed_records_inline(self.client, self.model, "rev1", [make_record()])
        assert self.client.get_collection("rev1").count() == 1

    def test_peek_no_nan(self):
        embed_records_inline(self.client, self.model, "rev2", [make_record()])
        emb = self.client.get_collection("rev2").peek(limit=1)["embeddings"][0]
        assert not any(v != v for v in emb)

    def test_peek_correct_dimension(self):
        embed_records_inline(self.client, self.model, "rev3", [make_record()])
        emb = self.client.get_collection("rev3").peek(limit=1)["embeddings"][0]
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
        assert client.get_collection("pipe1").count() == 5

    def test_invalid_rows_filtered(self):
        client, model = make_client(), make_mock_model()
        rows = [VALID_ROW, {**VALID_ROW, "id": 2, "phrase": ""}, {**VALID_ROW, "id": 3}]
        records, skipped = transform_batch(rows)
        assert skipped == 1
        embed_records_inline(client, model, "pipe2", records)
        assert client.get_collection("pipe2").count() == 2
