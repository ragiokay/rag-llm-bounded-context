"""
Test Plan — Causal Relation Transformer Module
Module under test: embedding/causal_transform.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import pytest
from pydantic import ValidationError
from causal_transform import CausalRelationRecord, transform_row, transform_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ROW = {
    "id": 1,
    "phrase": "Sales declined after the new product launch failed.",
    "question": "Did the failed launch cause a revenue drop?",
    "answer": "yes",
    "category": "cause",
    "domain": "Finance",
}

VALID_ROW_NO_ANSWER = {**VALID_ROW, "answer": "no"}


# ---------------------------------------------------------------------------
# Unit: transform_row — happy paths
# ---------------------------------------------------------------------------

class TestTransformRowHappyPath:
    def test_returns_causal_relation_record(self):
        record = transform_row(VALID_ROW)
        assert isinstance(record, CausalRelationRecord)

    def test_id_is_string(self):
        record = transform_row(VALID_ROW)
        assert record.id == "1"

    def test_cause_equals_phrase(self):
        record = transform_row(VALID_ROW)
        assert record.cause == VALID_ROW["phrase"]

    def test_consequence_encodes_yes_polarity(self):
        record = transform_row(VALID_ROW)
        assert record.consequence.startswith("does —")

    def test_consequence_encodes_no_polarity(self):
        record = transform_row(VALID_ROW_NO_ANSWER)
        assert record.consequence.startswith("does not —")

    def test_bounded_context_equals_domain(self):
        record = transform_row(VALID_ROW)
        assert record.bounded_context == "Finance"

    def test_embed_text_contains_phrase_and_question(self):
        record = transform_row(VALID_ROW)
        assert VALID_ROW["phrase"] in record.embed_text
        assert VALID_ROW["question"] in record.embed_text

    def test_source_phrase_preserved(self):
        record = transform_row(VALID_ROW)
        assert record.source_phrase == VALID_ROW["phrase"]


# ---------------------------------------------------------------------------
# Unit: transform_row — boundary & failure modes
# ---------------------------------------------------------------------------

class TestTransformRowBoundary:
    def test_missing_phrase_returns_none(self):
        assert transform_row({**VALID_ROW, "phrase": ""}) is None

    def test_missing_question_returns_none(self):
        assert transform_row({**VALID_ROW, "question": ""}) is None

    def test_missing_domain_returns_none(self):
        assert transform_row({**VALID_ROW, "domain": ""}) is None

    def test_whitespace_only_phrase_returns_none(self):
        assert transform_row({**VALID_ROW, "phrase": "   "}) is None

    def test_missing_category_defaults_to_unknown(self):
        record = transform_row({**VALID_ROW, "category": ""})
        assert record is not None
        assert record.category == "unknown"

    def test_none_values_do_not_raise(self):
        row = {k: None for k in VALID_ROW}
        result = transform_row(row)
        assert result is None

    def test_extra_keys_are_ignored(self):
        row = {**VALID_ROW, "unexpected_column": "foo"}
        record = transform_row(row)
        assert record is not None


# ---------------------------------------------------------------------------
# Unit: CausalRelationRecord — schema validation
# ---------------------------------------------------------------------------

class TestCausalRelationRecordSchema:
    def test_empty_cause_raises(self):
        with pytest.raises(ValidationError):
            CausalRelationRecord(
                id="1", cause="", consequence="x", bounded_context="Finance",
                category="cause", source_phrase="x", embed_text="x"
            )

    def test_empty_embed_text_raises(self):
        with pytest.raises(ValidationError):
            CausalRelationRecord(
                id="1", cause="x", consequence="x", bounded_context="Finance",
                category="cause", source_phrase="x", embed_text=""
            )

    def test_to_chroma_metadata_keys(self):
        record = transform_row(VALID_ROW)
        meta = record.to_chroma_metadata()
        assert set(meta.keys()) == {"cause", "consequence", "bounded_context", "category", "source_phrase"}

    def test_to_chroma_metadata_no_embed_text(self):
        record = transform_row(VALID_ROW)
        assert "embed_text" not in record.to_chroma_metadata()


# ---------------------------------------------------------------------------
# Unit: transform_batch
# ---------------------------------------------------------------------------

class TestTransformBatch:
    def test_all_valid_returns_correct_count(self):
        rows = [VALID_ROW, {**VALID_ROW, "id": 2}]
        records, skipped = transform_batch(rows)
        assert len(records) == 2
        assert skipped == 0

    def test_mixed_valid_invalid_counts_skipped(self):
        rows = [VALID_ROW, {**VALID_ROW, "phrase": ""}, {**VALID_ROW, "id": 3}]
        records, skipped = transform_batch(rows)
        assert len(records) == 2
        assert skipped == 1

    def test_empty_input_returns_empty(self):
        records, skipped = transform_batch([])
        assert records == []
        assert skipped == 0

    def test_all_invalid_returns_all_skipped(self):
        bad = {"id": 1, "phrase": "", "question": "", "answer": "yes", "category": "", "domain": ""}
        records, skipped = transform_batch([bad, bad])
        assert len(records) == 0
        assert skipped == 2

    def test_large_batch_does_not_raise(self):
        rows = [{**VALID_ROW, "id": i} for i in range(500)]
        records, skipped = transform_batch(rows)
        assert len(records) == 500
        assert skipped == 0
