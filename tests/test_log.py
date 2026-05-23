import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedding.seed_log import parse_domain_events, parse_commands

# Minimal realistic log snippets copied from the actual log format
_DOMAIN_EVENTS_FIXTURE = """
[2026-05-12 00:05:12][DEBUG] [Domain Events] ===========================
[2026-05-12 00:05:12][DEBUG] +-----------------------------------------------+-------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | Domain Event                                  | Aggregate State   | UC ID   | Sentence                                                                                                 |
                     +===============================================+===================+=========+==========================================================================================================+
                     | user logged in                                | True              | UC-001  | The user is logged in to the system .                                                                    |
                     +-----------------------------------------------+-------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | user verified                                 | False             | UC-001  | The system verifies the user .                                                                           |
                     +-----------------------------------------------+-------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | meeting scheduled                             | True              | UC-005  | The initiated meeting is scheduled and the system sends new meeting messages to all participants .       |
                     +-----------------------------------------------+-------------------+---------+----------------------------------------------------------------------------------------------------------+
[2026-05-12 00:05:12]-----------------------------------------------------------
[2026-05-12 00:05:12] [Strategic Design - Step3]: Pair Commands with Events.
[2026-05-12 00:05:12]-----------------------------------------------------------
"""

_COMMANDS_FIXTURE = """
[2026-05-12 00:05:47]-----------------------------------------------------------
[2026-05-12 00:05:47] [Strategic Design - Step4]: Assign Actors For Commands
[2026-05-12 00:05:47]-----------------------------------------------------------
[2026-05-12 00:05:49][DEBUG] [Commands] ===========================
[2026-05-12 00:05:49][DEBUG] +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | Command                                      | Actor               | UC ID   | Sentence                                                                                                 |
                     +==============================================+=====================+=========+==========================================================================================================+
                     | login                                        | name: user          | UC-001  | The User enters the user's username and password and presses the " LOGIN " button .                      |
                     +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | initiate meeting                             | name: initiator     | UC-005  | The user clicks " INITIATE_MEETING " button .                                                            |
                     +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | cancel meeting                               | name: initiator     | UC-011  | The user clicks " CANCEL_MEETING " .                                                                     |
                     +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | register user                                | None                | UC-002  | User is registered in the system and has the username and password to access the system .                |
                     +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
                     | get meeting list data                        | name: system        | UC-012  | The system sends a request to the backend to get meeting list data .                                     |
                     +----------------------------------------------+---------------------+---------+----------------------------------------------------------------------------------------------------------+
[2026-05-12 00:05:49]-----------------------------------------------------------
[2026-05-12 00:05:49] [Strategic Design - Step5]: Create Policies Between Events And Commands
[2026-05-12 00:05:49]-----------------------------------------------------------
"""


class TestParseDomainEvents:
    def test_count(self):
        records = parse_domain_events(_DOMAIN_EVENTS_FIXTURE)
        assert len(records) == 3

    def test_fields_present(self):
        records = parse_domain_events(_DOMAIN_EVENTS_FIXTURE)
        for r in records:
            assert "domain_event" in r
            assert "document" in r
            assert "source_phrase" in r

    def test_event_names(self):
        records = parse_domain_events(_DOMAIN_EVENTS_FIXTURE)
        names = [r["domain_event"] for r in records]
        assert "user logged in" in names
        assert "user verified" in names
        assert "meeting scheduled" in names

    def test_embed_text_is_sentence(self):
        records = parse_domain_events(_DOMAIN_EVENTS_FIXTURE)
        verified = next(r for r in records if r["domain_event"] == "user verified")
        assert "verifies the user" in verified["document"]

    def test_no_empty_embed_text(self):
        records = parse_domain_events(_DOMAIN_EVENTS_FIXTURE)
        assert all(r["document"].strip() for r in records)

    def test_empty_log_returns_empty(self):
        assert parse_domain_events("") == []

    def test_missing_step3_marker_still_parses(self):
        # Without the Step3 boundary marker the parser should still return rows
        text = _DOMAIN_EVENTS_FIXTURE.replace(
            "[Strategic Design - Step3]: Pair Commands with Events.", ""
        )
        records = parse_domain_events(text)
        assert len(records) == 3


class TestParseCommands:
    def test_count(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        assert len(records) == 5

    def test_fields_present(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        for r in records:
            assert "command" in r
            assert "document" in r
            assert "source_phrase" in r

    def test_actor_parsed(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        login = next(r for r in records if r["command"] == "login")
        assert login.get("user_roles") == "user"

    def test_actor_none_omitted(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        register = next(r for r in records if r["command"] == "register user")
        assert "user_roles" not in register

    def test_actor_system(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        get_list = next(r for r in records if r["command"] == "get meeting list data")
        assert get_list.get("user_roles") == "system"

    def test_embed_text_is_sentence(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        initiate = next(r for r in records if r["command"] == "initiate meeting")
        assert "INITIATE_MEETING" in initiate["document"]

    def test_no_empty_embed_text(self):
        records = parse_commands(_COMMANDS_FIXTURE)
        assert all(r["document"].strip() for r in records)

    def test_missing_step4_marker_returns_empty(self):
        text = _COMMANDS_FIXTURE.replace(
            "[Strategic Design - Step4]: Assign Actors For Commands", ""
        )
        assert parse_commands(text) == []

    def test_empty_log_returns_empty(self):
        assert parse_commands("") == []
