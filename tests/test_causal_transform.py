"""
Test Plan — Causal Relation Transformer Module
Module under test: embedding/causal_transform.py
DDD fields: domain_event, command, policy, aggregate, bounded_context
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import pytest
from pydantic import ValidationError
from causal_transform import CausalRelationRecord, transform_row, transform_batch


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
        assert transform_row(VALID_ROW).id == "1"

    def test_domain_event_equals_phrase(self):
        assert transform_row(VALID_ROW).domain_event == VALID_ROW["phrase"]

    def test_command_equals_question(self):
        assert transform_row(VALID_ROW).command == VALID_ROW["question"]

    def test_policy_encodes_yes_polarity(self):
        assert transform_row(VALID_ROW).policy.startswith("does —")

    def test_policy_encodes_no_polarity(self):
        assert transform_row(VALID_ROW_NO_ANSWER).policy.startswith("does not —")

    def test_aggregate_equals_category(self):
        assert transform_row(VALID_ROW).aggregate == "cause"

    def test_bounded_context_equals_domain(self):
        assert transform_row(VALID_ROW).bounded_context == "Finance"

    def test_embed_text_contains_phrase_and_question(self):
        record = transform_row(VALID_ROW)
        assert VALID_ROW["phrase"] in record.embed_text
        assert VALID_ROW["question"] in record.embed_text

    def test_source_phrase_preserved(self):
        assert transform_row(VALID_ROW).source_phrase == VALID_ROW["phrase"]

    def test_optional_fields_are_none_by_default(self):
        record = transform_row(VALID_ROW)
        assert record.views is None
        assert record.user_roles is None
        assert record.process is None


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
        assert record.aggregate == "unknown"

    def test_none_values_do_not_raise(self):
        assert transform_row({k: None for k in VALID_ROW}) is None

    def test_extra_keys_are_ignored(self):
        assert transform_row({**VALID_ROW, "unexpected_column": "foo"}) is not None


# ---------------------------------------------------------------------------
# Unit: CausalRelationRecord — schema validation
# ---------------------------------------------------------------------------

class TestCausalRelationRecordSchema:
    def test_empty_domain_event_raises(self):
        with pytest.raises(ValidationError):
            CausalRelationRecord(
                id="1", domain_event="", command="x", policy="x",
                aggregate="cause", bounded_context="Finance",
                source_phrase="x", embed_text="x"
            )

    def test_empty_embed_text_raises(self):
        with pytest.raises(ValidationError):
            CausalRelationRecord(
                id="1", domain_event="x", command="x", policy="x",
                aggregate="cause", bounded_context="Finance",
                source_phrase="x", embed_text=""
            )

    def test_to_chroma_metadata_required_keys(self):
        meta = transform_row(VALID_ROW).to_chroma_metadata()
        assert set(meta.keys()) == {
            "domain_event", "command", "policy",
            "aggregate", "bounded_context", "source_phrase"
        }

    def test_to_chroma_metadata_excludes_embed_text(self):
        assert "embed_text" not in transform_row(VALID_ROW).to_chroma_metadata()

    def test_to_chroma_metadata_excludes_none_optional_fields(self):
        meta = transform_row(VALID_ROW).to_chroma_metadata()
        assert "views" not in meta
        assert "user_roles" not in meta
        assert "process" not in meta

    def test_optional_fields_included_when_set(self):
        record = transform_row(VALID_ROW)
        record.views = "sales dashboard"
        meta = record.to_chroma_metadata()
        assert meta["views"] == "sales dashboard"


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
        assert records == [] and skipped == 0

    def test_all_invalid_returns_all_skipped(self):
        bad = {"id": 1, "phrase": "", "question": "", "answer": "yes", "category": "", "domain": ""}
        records, skipped = transform_batch([bad, bad])
        assert len(records) == 0 and skipped == 2

    def test_large_batch_does_not_raise(self):
        rows = [{**VALID_ROW, "id": i} for i in range(500)]
        records, skipped = transform_batch(rows)
        assert len(records) == 500 and skipped == 0
