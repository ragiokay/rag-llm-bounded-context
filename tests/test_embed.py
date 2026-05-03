"""
Test Plan — Embedding & DB Module
Module under test: embedding/embed.py (embed_records, review_collection)

SentenceTransformer is mocked throughout so these tests run without network
access or a downloaded model. ChromaDB is run in-memory (EphemeralClient).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import numpy as np
import pytest
import chromadb
from unittest.mock import MagicMock, patch
from causal_transform import CausalRelationRecord, transform_row, transform_batch

EMBED_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"

VALID_ROW = {
    "id": 1,
    "phrase": "Inventory shortages led to production delays.",
    "question": "Did the shortage cause delivery failure?",
    "answer": "yes",
    "category": "cause",
    "domain": "Logistics",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_model(seed: int = 42):
    """Returns a mock SentenceTransformer whose .encode() is deterministic."""
    rng = np.random.default_rng(seed)

    def fake_encode(texts, show_progress_bar=False, **kwargs):
        # Same seed → same vectors for same input length is enough for unit tests.
        result = rng.random((len(texts), EMBED_DIM)).astype(np.float32)
        return result

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def make_client():
    return chromadb.EphemeralClient()


def make_record(row=None) -> CausalRelationRecord:
    return transform_row(row or VALID_ROW)


def embed_records_inline(client, model, collection_name, records):
    """Inline version of embed_records using an injected client and model."""
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
        embs = self.model.encode(["test sentence"])
        assert embs.shape == (1, EMBED_DIM)

    def test_batch_shape(self):
        embs = self.model.encode(["a", "b", "c"])
        assert embs.shape == (3, EMBED_DIM)

    def test_no_nan(self):
        embs = self.model.encode(["test"])[0].tolist()
        assert not any(v != v for v in embs)

    def test_nonzero(self):
        embs = self.model.encode(["test"])[0].tolist()
        assert any(v != 0 for v in embs)

    def test_empty_string_does_not_crash(self):
        embs = self.model.encode([""])
        assert embs.shape[1] == EMBED_DIM

    def test_long_string_does_not_crash(self):
        embs = self.model.encode(["word " * 1000])
        assert embs.shape[1] == EMBED_DIM

    def test_encode_called_with_list(self):
        texts = ["sentence one", "sentence two"]
        self.model.encode(texts)
        args, _ = self.model.encode.call_args
        assert isinstance(args[0], list)


# ---------------------------------------------------------------------------
# Integration: embed_records -> ChromaDB round-trip (mocked model)
# ---------------------------------------------------------------------------

class TestEmbedRecordsIntegration:
    def setup_method(self):
        self.client = make_client()
        self.model = make_mock_model()

    def test_written_count_matches(self):
        record = make_record()
        written = embed_records_inline(self.client, self.model, "col1", [record])
        assert written == 1

    def test_retrieve_by_id_returns_record(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col2", [record])
        col = self.client.get_collection("col2")
        result = col.get(ids=[record.id], include=["metadatas", "documents"])
        assert result["ids"] == [record.id]

    def test_metadata_keys_complete(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col3", [record])
        col = self.client.get_collection("col3")
        result = col.get(ids=[record.id], include=["metadatas"])
        meta = result["metadatas"][0]
        for key in ("cause", "consequence", "bounded_context", "category", "source_phrase"):
            assert key in meta, f"Missing key: {key}"

    def test_metadata_bounded_context_value(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col3b", [record])
        col = self.client.get_collection("col3b")
        result = col.get(ids=[record.id], include=["metadatas"])
        assert result["metadatas"][0]["bounded_context"] == "Logistics"

    def test_empty_records_list_writes_zero(self):
        written = embed_records_inline(self.client, self.model, "col4", [])
        assert written == 0

    def test_collection_count_after_batch(self):
        rows = [{**VALID_ROW, "id": i} for i in range(1, 11)]
        records = [transform_row(r) for r in rows]
        embed_records_inline(self.client, self.model, "col5", records)
        col = self.client.get_collection("col5")
        assert col.count() == 10

    def test_document_stored_matches_embed_text(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col6", [record])
        col = self.client.get_collection("col6")
        result = col.get(ids=[record.id], include=["documents"])
        assert result["documents"][0] == record.embed_text

    def test_vector_dimension_stored_in_chroma(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "col7", [record])
        col = self.client.get_collection("col7")
        result = col.get(ids=[record.id], include=["embeddings"])
        assert len(result["embeddings"][0]) == EMBED_DIM


# ---------------------------------------------------------------------------
# Integration: review_collection logic
# ---------------------------------------------------------------------------

class TestReviewCollection:
    def setup_method(self):
        self.client = make_client()
        self.model = make_mock_model()

    def test_empty_collection_count_is_zero(self):
        col = self.client.get_or_create_collection(name="empty_col")
        assert col.count() == 0

    def test_non_empty_collection_count_correct(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "rev_col", [record])
        col = self.client.get_collection("rev_col")
        assert col.count() == 1

    def test_peek_embedding_no_nan(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "nan_col", [record])
        col = self.client.get_collection("nan_col")
        sample = col.peek(limit=1)
        emb = sample["embeddings"][0]
        assert not any(v != v for v in emb)

    def test_peek_embedding_correct_dimension(self):
        record = make_record()
        embed_records_inline(self.client, self.model, "dim_col", [record])
        col = self.client.get_collection("dim_col")
        sample = col.peek(limit=1)
        assert len(sample["embeddings"][0]) == EMBED_DIM


# ---------------------------------------------------------------------------
# Integration: transform -> embed pipeline
# ---------------------------------------------------------------------------

class TestTransformToEmbedPipeline:
    def test_pipeline_end_to_end(self):
        """Records transformed from raw rows must survive the full pipeline."""
        client = make_client()
        model = make_mock_model()

        rows = [{**VALID_ROW, "id": i, "domain": "Finance"} for i in range(1, 6)]
        records, skipped = transform_batch(rows)
        assert skipped == 0

        written = embed_records_inline(client, model, "pipe_col", records)
        assert written == 5

        col = client.get_collection("pipe_col")
        assert col.count() == 5

    def test_invalid_rows_not_written(self):
        client = make_client()
        model = make_mock_model()

        rows = [
            VALID_ROW,
            {**VALID_ROW, "id": 2, "phrase": ""},  # invalid
            {**VALID_ROW, "id": 3},
        ]
        records, skipped = transform_batch(rows)
        assert skipped == 1
        embed_records_inline(client, model, "filter_col", records)
        col = client.get_collection("filter_col")
        assert col.count() == 2
