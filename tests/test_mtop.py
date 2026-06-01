"""
Test Plan — MTOP Seed Module
Module under test: embedding/seed_mtop.py
No network access needed — all tests use inline fixture rows.
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
COLLECTION_NAME = "test_mtop"


def make_mock_model(seed=42):
    rng = np.random.default_rng(seed)
    def fake_encode(texts, show_progress_bar=False, **kwargs):
        return rng.random((len(texts), EMBED_DIM)).astype(np.float32)
    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


def _uuid(id_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


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
        assert parse_record(make_row()) is not None

    def test_returns_dict(self):
        assert isinstance(parse_record(make_row()), dict)

    def test_command_field_is_normalized(self):
        record = parse_record(make_row())
        assert record["command"] == "Create Reminder"

    def test_policy_contains_intent(self):
        record = parse_record(make_row())
        assert "create_reminder" in record["policy"]

    def test_aggregate_is_object_part(self):
        record = parse_record(make_row())
        assert record["aggregate"] == "reminder"

    def test_bounded_context_is_mtop(self):
        record = parse_record(make_row())
        assert record["bounded_context"] == "MTOP"

    def test_document_is_utterance(self):
        row = make_row()
        record = parse_record(row)
        assert record["document"] == row["text"]

    def test_trigger_span_present(self):
        record = parse_record(make_row())
        assert record.get("trigger_span") is not None

    def test_in_prefix_in_label_text_handled(self):
        row = make_row(label_text="IN:CREATE_REMINDER")
        record = parse_record(row)
        assert record is not None
        assert record["command"] == "Create Reminder"

    def test_send_message(self):
        row = make_row(
            text="text Matthew and Helen that are you free",
            label_text="SEND_MESSAGE",
        )
        record = parse_record(row)
        assert record is not None
        assert record["command"] == "Send Message"

    def test_delete_alarm(self):
        row = make_row(text="Delete my 7am alarm", label_text="DELETE_ALARM")
        record = parse_record(row)
        assert record is not None
        assert record["aggregate"] == "alarm"


# ---------------------------------------------------------------------------
# Unit: parse_record — filtering / failure modes
# ---------------------------------------------------------------------------

class TestParseRecordFiltering:
    def test_query_intent_returns_none(self):
        row = make_row(text="What's the weather in New Zealand?", label_text="GET_WEATHER")
        assert parse_record(row) is None

    def test_question_intent_returns_none(self):
        row = make_row(text="Any news today?", label_text="QUESTION_NEWS")
        assert parse_record(row) is None

    def test_is_true_intent_returns_none(self):
        row = make_row(text="Is this recipe vegetarian?", label_text="IS_TRUE_RECIPES")
        assert parse_record(row) is None

    def test_empty_text_returns_none(self):
        row = {"text": "", "label_text": "CREATE_REMINDER"}
        assert parse_record(row) is None

    def test_empty_intent_returns_none(self):
        row = {"text": "Remind me something", "label_text": ""}
        assert parse_record(row) is None

    def test_missing_fields_returns_none(self):
        assert parse_record({}) is None

    def test_utterance_field_alias_accepted(self):
        row = {"utterance": "Set a timer for 5 minutes", "label_text": "CREATE_TIMER"}
        assert parse_record(row) is not None

    def test_intent_field_alias_accepted(self):
        row = {"text": "Set a timer for 5 minutes", "intent": "CREATE_TIMER"}
        assert parse_record(row) is not None


# ---------------------------------------------------------------------------
# Integration: embed_and_store -> Qdrant round-trip
# ---------------------------------------------------------------------------

def _make_records():
    rows = [
        make_row("Remind me to take medicine at 9am", "CREATE_REMINDER"),
        make_row("Send a message to Alice saying I am late", "SEND_MESSAGE"),
        make_row("Delete my morning alarm", "DELETE_ALARM"),
    ]
    return [r for r in (parse_record(row) for row in rows) if r]


class TestEmbedAndStore:
    def setup_method(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self.model = make_mock_model()
        self.records = _make_records()
        self._client = self.client

    def _run_embed_and_store(self, records=None, col=COLLECTION_NAME):
        import seed_mtop
        orig = seed_mtop.QdrantClient

        def fake_client(url=None, **kwargs):
            return self._client

        seed_mtop.QdrantClient = fake_client
        try:
            result = embed_and_store(records if records is not None else self.records,
                                     col, self.model)
        finally:
            seed_mtop.QdrantClient = orig
        return result

    def test_all_records_created(self):
        assert len(self.records) == 3

    def test_written_count_matches(self):
        written = self._run_embed_and_store()
        assert written == len(self.records)

    def test_empty_records_writes_zero(self):
        assert self._run_embed_and_store(records=[]) == 0

    def test_retrieve_by_document(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        assert len(results) == 1

    def test_metadata_has_command(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        meta = results[0].payload
        assert "command" in meta
        assert meta["command"] == "Create Reminder"

    def test_metadata_has_policy(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        meta = results[0].payload
        assert "policy" in meta
        assert "create_reminder" in meta["policy"]

    def test_metadata_has_trigger_span(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        assert "trigger_span" in results[0].payload

    def test_metadata_has_bounded_context(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        assert results[0].payload["bounded_context"] == "MTOP"

    def test_vector_dimension(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_vectors=True,
        )
        assert len(results[0].vector) == EMBED_DIM

    def test_document_matches_text(self):
        self._run_embed_and_store()
        record = self.records[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, record["document"][:120]))
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
        assert results[0].payload["document"] == record["document"]

    def test_ids_are_unique(self):
        self._run_embed_and_store()
        points, _ = self.client.scroll(collection_name=COLLECTION_NAME, limit=100)
        ids = [str(p.id) for p in points]
        assert len(ids) == len(set(ids))
