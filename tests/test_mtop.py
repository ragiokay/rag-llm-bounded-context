"""
Test Plan — MTOP Seed Module
Module under test: embedding/seed_mtop.py
No network access needed — all tests use inline fixture rows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding"))

import pytest
import chromadb
import numpy as np
from unittest.mock import MagicMock
from causal_transform import CausalRelationRecord
from seed_mtop import (
    classify_mtop_intent,
    intent_to_command_name,
    intent_to_aggregate,
    extract_trigger_span,
    parse_record,
    embed_and_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_row(text="Remind me to start cooking dinner in 10 minutes",
             label_text="CREATE_REMINDER"):
    return {"text": text, "label_text": label_text}


EMBED_DIM = 384


def make_mock_model(seed=42):
    rng = np.random.default_rng(seed)
    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)
    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


# ---------------------------------------------------------------------------
# Unit: classify_mtop_intent
# ---------------------------------------------------------------------------

class TestClassifyMtopIntent:
    def test_create_is_command(self):
        assert classify_mtop_intent("CREATE_REMINDER") == "command"

    def test_delete_is_command(self):
        assert classify_mtop_intent("DELETE_ALARM") == "command"

    def test_update_is_command(self):
        assert classify_mtop_intent("UPDATE_CALL") == "command"

    def test_send_is_command(self):
        assert classify_mtop_intent("SEND_MESSAGE") == "command"

    def test_set_is_command(self):
        assert classify_mtop_intent("SET_AVAILABLE") == "command"

    def test_play_is_command(self):
        assert classify_mtop_intent("PLAY_MUSIC") == "command"

    def test_get_is_query(self):
        assert classify_mtop_intent("GET_WEATHER") == "query"

    def test_question_is_query(self):
        assert classify_mtop_intent("QUESTION_NEWS") == "query"

    def test_is_true_is_query(self):
        assert classify_mtop_intent("IS_TRUE_RECIPES") == "query"

    def test_unknown_is_review(self):
        assert classify_mtop_intent("PREFER") == "review"

    def test_in_prefix_stripped(self):
        assert classify_mtop_intent("IN:CREATE_REMINDER") == "command"
        assert classify_mtop_intent("IN:GET_WEATHER") == "query"


# ---------------------------------------------------------------------------
# Unit: intent_to_command_name
# ---------------------------------------------------------------------------

class TestIntentToCommandName:
    def test_create_reminder(self):
        assert intent_to_command_name("CREATE_REMINDER") == "Create Reminder"

    def test_send_message(self):
        assert intent_to_command_name("SEND_MESSAGE") == "Send Message"

    def test_delete_alarm(self):
        assert intent_to_command_name("DELETE_ALARM") == "Delete Alarm"

    def test_in_prefix_stripped(self):
        assert intent_to_command_name("IN:CREATE_ALARM") == "Create Alarm"

    def test_multi_word_object(self):
        assert intent_to_command_name("CREATE_HOME_ASSISTANT") == "Create Home Assistant"


# ---------------------------------------------------------------------------
# Unit: intent_to_aggregate
# ---------------------------------------------------------------------------

class TestIntentToAggregate:
    def test_reminder(self):
        assert intent_to_aggregate("CREATE_REMINDER") == "REMINDER"

    def test_message(self):
        assert intent_to_aggregate("SEND_MESSAGE") == "MESSAGE"

    def test_in_prefix_stripped(self):
        assert intent_to_aggregate("IN:DELETE_ALARM") == "ALARM"

    def test_no_underscore_returns_full(self):
        assert intent_to_aggregate("PREFER") == "PREFER"


# ---------------------------------------------------------------------------
# Unit: extract_trigger_span
# ---------------------------------------------------------------------------

class TestExtractTriggerSpan:
    def test_imperative_verb_start(self):
        assert extract_trigger_span("Set an alarm for Thursday at 6") == "Set an alarm"

    def test_remind_me(self):
        assert extract_trigger_span("Remind me to start cooking dinner in 10 minutes") == "Remind me"

    def test_please_skipped(self):
        span = extract_trigger_span("Please send a message to Bob")
        assert span.startswith("send")

    def test_send_verb(self):
        assert extract_trigger_span("text Matthew and Helen") == "text Matthew"

    def test_empty_text(self):
        assert extract_trigger_span("") == ""

    def test_single_word(self):
        assert extract_trigger_span("Cancel") == "Cancel"

    def test_stops_at_for(self):
        assert extract_trigger_span("Create a reminder for tomorrow") == "Create a reminder"


# ---------------------------------------------------------------------------
# Unit: parse_record — happy paths
# ---------------------------------------------------------------------------

class TestParseRecordHappyPath:
    def test_returns_record_for_command(self):
        row = make_row()
        assert parse_record(row, 0) is not None

    def test_returns_causal_relation_record(self):
        row = make_row()
        assert isinstance(parse_record(row, 0), CausalRelationRecord)

    def test_command_field_is_normalized(self):
        row = make_row()
        record = parse_record(row, 0)
        assert record.command == "Create Reminder"

    def test_policy_contains_intent(self):
        row = make_row()
        record = parse_record(row, 0)
        assert "create_reminder" in record.policy

    def test_aggregate_is_object_part(self):
        row = make_row()
        record = parse_record(row, 0)
        assert record.aggregate == "reminder"

    def test_bounded_context_is_mtop(self):
        row = make_row()
        record = parse_record(row, 0)
        assert record.bounded_context == "MTOP"

    def test_embed_text_is_utterance(self):
        row = make_row()
        record = parse_record(row, 0)
        assert record.embed_text == row["text"]

    def test_trigger_span_not_none(self):
        row = make_row()
        record = parse_record(row, 0)
        assert record.trigger_span is not None

    def test_id_contains_index(self):
        row = make_row()
        record = parse_record(row, 42)
        assert "42" in record.id

    def test_in_prefix_in_label_text_handled(self):
        row = make_row(label_text="IN:CREATE_REMINDER")
        record = parse_record(row, 0)
        assert record is not None
        assert record.command == "Create Reminder"

    def test_send_message(self):
        row = make_row(
            text="text Matthew and Helen that are you free",
            label_text="SEND_MESSAGE",
        )
        record = parse_record(row, 1)
        assert record is not None
        assert record.command == "Send Message"

    def test_delete_alarm(self):
        row = make_row(text="Delete my 7am alarm", label_text="DELETE_ALARM")
        record = parse_record(row, 2)
        assert record is not None
        assert record.aggregate == "alarm"


# ---------------------------------------------------------------------------
# Unit: parse_record — filtering / failure modes
# ---------------------------------------------------------------------------

class TestParseRecordFiltering:
    def test_query_intent_returns_none(self):
        row = make_row(text="What's the weather in New Zealand?", label_text="GET_WEATHER")
        assert parse_record(row, 0) is None

    def test_question_intent_returns_none(self):
        row = make_row(text="Any news today?", label_text="QUESTION_NEWS")
        assert parse_record(row, 0) is None

    def test_is_true_intent_returns_none(self):
        row = make_row(text="Is this recipe vegetarian?", label_text="IS_TRUE_RECIPES")
        assert parse_record(row, 0) is None

    def test_empty_text_returns_none(self):
        row = {"text": "", "label_text": "CREATE_REMINDER"}
        assert parse_record(row, 0) is None

    def test_empty_intent_returns_none(self):
        row = {"text": "Remind me something", "label_text": ""}
        assert parse_record(row, 0) is None

    def test_missing_fields_returns_none(self):
        assert parse_record({}, 0) is None

    def test_utterance_field_alias_accepted(self):
        row = {"utterance": "Set a timer for 5 minutes", "label_text": "CREATE_TIMER"}
        record = parse_record(row, 0)
        assert record is not None

    def test_intent_field_alias_accepted(self):
        row = {"text": "Set a timer for 5 minutes", "intent": "CREATE_TIMER"}
        record = parse_record(row, 0)
        assert record is not None


# ---------------------------------------------------------------------------
# Integration: embed_and_store -> ChromaDB round-trip
# ---------------------------------------------------------------------------

def _make_records():
    rows = [
        make_row("Remind me to take medicine at 9am", "CREATE_REMINDER"),
        make_row("Send a message to Alice saying I am late", "SEND_MESSAGE"),
        make_row("Delete my morning alarm", "DELETE_ALARM"),
    ]
    return [r for r in (parse_record(row, i) for i, row in enumerate(rows)) if r]


class TestEmbedAndStore:
    def setup_method(self):
        self.client = chromadb.EphemeralClient()
        self.model = make_mock_model()
        self.col = self.client.get_or_create_collection("test_mtop")
        self.records = _make_records()

    def test_all_records_created(self):
        assert len(self.records) == 3

    def test_written_count_matches(self):
        written = embed_and_store(self.records, self.col, self.model)
        assert written == len(self.records)

    def test_empty_records_writes_zero(self):
        assert embed_and_store([], self.col, self.model) == 0

    def test_retrieve_by_id(self):
        embed_and_store(self.records, self.col, self.model)
        result = self.col.get(ids=[self.records[0].id])
        assert result["ids"] == [self.records[0].id]

    def test_metadata_has_command(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert "command" in meta
        assert meta["command"] == "Create Reminder"

    def test_metadata_has_policy(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert "policy" in meta
        assert "create_reminder" in meta["policy"]

    def test_metadata_has_trigger_span(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert "trigger_span" in meta

    def test_metadata_has_bounded_context(self):
        embed_and_store(self.records, self.col, self.model)
        meta = self.col.get(
            ids=[self.records[0].id], include=["metadatas"]
        )["metadatas"][0]
        assert meta["bounded_context"] == "MTOP"

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

    def test_ids_are_unique(self):
        ids = [r.id for r in self.records]
        assert len(ids) == len(set(ids))
