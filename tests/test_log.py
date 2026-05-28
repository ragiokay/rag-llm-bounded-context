import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedding.seed_log import parse_domain_events, parse_commands, parse_command_event_pairs, parse_policies

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


_PAIRS_FIXTURE = """
[2026-05-12 00:05:12]-----------------------------------------------------------
[2026-05-12 00:05:12] [Strategic Design - Step3]: Pair Commands with Events.
[2026-05-12 00:05:12]-----------------------------------------------------------
[2026-05-12 00:05:14][DEBUG] Event-Command found: "user verified"->"login"
                      sentence: The User enters the user's username and password and presses the " LOGIN " button . The system verifies the user .
                      Causal prediction confidence: 0.9952837824821472
[2026-05-12 00:05:16][DEBUG] Event-Command found: "meeting initiated"->"initiate meeting"
                      sentence: The user clicks " INITIATE_MEETING " button . The system sends a request to the backend to initiate a meeting .
                      Causal prediction confidence: 0.9998838901519775
[2026-05-12 00:05:21][DEBUG] Event-Command found: "meeting scheduled"->"establish agenda"
                      sentence: The user establishes the meeting 's agendas . The initiated meeting is scheduled and the system sends new meeting messages to all participants .
                      Causal prediction confidence: 0.9962913990020752
[2026-05-12 00:05:47]-----------------------------------------------------------
[2026-05-12 00:05:47] [Strategic Design - Step4]: Assign Actors For Commands
[2026-05-12 00:05:47]-----------------------------------------------------------
"""


class TestParseCommandEventPairs:
    def test_count(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        assert len(records) == 3

    def test_fields_present(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        for r in records:
            assert "domain_event" in r
            assert "command" in r
            assert "document" in r
            assert "source_phrase" in r
            assert "commands_events_pairs" in r

    def test_event_and_command_parsed(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        login = next(r for r in records if r["command"] == "login")
        assert login["domain_event"] == "user verified"

    def test_commands_events_pairs_is_list_of_list(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        for r in records:
            assert isinstance(r["commands_events_pairs"], list)
            assert isinstance(r["commands_events_pairs"][0], list)
            assert len(r["commands_events_pairs"][0]) == 2

    def test_pairs_order_is_command_then_event(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        login = next(r for r in records if r["command"] == "login")
        assert login["commands_events_pairs"] == [["login", "user verified"]]

    def test_document_is_compound_sentence(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        login = next(r for r in records if r["command"] == "login")
        assert "LOGIN" in login["document"]
        assert "verifies" in login["document"]

    def test_no_empty_document(self):
        records = parse_command_event_pairs(_PAIRS_FIXTURE)
        assert all(r["document"].strip() for r in records)

    def test_empty_log_returns_empty(self):
        assert parse_command_event_pairs("") == []

    def test_missing_step3_marker_returns_empty(self):
        text = _PAIRS_FIXTURE.replace(
            "[Strategic Design - Step3]: Pair Commands with Events.", ""
        )
        assert parse_command_event_pairs(text) == []


_POLICY_FIXTURE = """
[2026-05-12 00:05:49]-----------------------------------------------------------
[2026-05-12 00:05:49] [Strategic Design - Step5]: Create Policies Between Events And Commands
[2026-05-12 00:05:49]-----------------------------------------------------------
[2026-05-12 00:05:51][DEBUG] (Policy)Event-Command found: "user profile created"->"register user"
                      sentence: The system checks whether the username is taken and creates a user profile for the user . User is registered in the system and has the username and password to access the system .
                      Causal prediction confidence: 0.558683454990387
[2026-05-12 00:06:35][DEBUG] (Policy)Event-Command found: "meeting initiated"->"get meeting list data"
                      sentence: The user is initiating a meeting or managing a meeting. The system sends a request to the backend to get meeting list data .
                      Causal prediction confidence: 0.9081053733825684
[2026-05-12 00:07:10][DEBUG] (Policy)Event-Command found: "administrator logged in"->"delete user level"
                      sentence: The administrator is logged in to the system. The user can add , edit or delete the user levels .
                      Causal prediction confidence: 0.881516695022583
[2026-05-12 00:07:10]-----------------------------------------------------------
[2026-05-12 00:07:10] [Strategic Design - Step6]: Group Event-Command Pairs Into Aggregates
[2026-05-12 00:07:10]-----------------------------------------------------------
"""


class TestParsePolicies:
    def test_count(self):
        records = parse_policies(_POLICY_FIXTURE)
        assert len(records) == 3

    def test_fields_present(self):
        records = parse_policies(_POLICY_FIXTURE)
        for r in records:
            assert "policy" in r
            assert "document" in r

    def test_no_extra_fields(self):
        records = parse_policies(_POLICY_FIXTURE)
        for r in records:
            assert set(r.keys()) == {"policy", "document"}

    def test_policy_value_format(self):
        records = parse_policies(_POLICY_FIXTURE)
        admin = next(r for r in records if "administrator" in r["policy"])
        assert admin["policy"] == '"administrator logged in"->"delete user level"'

    def test_document_is_sentence(self):
        records = parse_policies(_POLICY_FIXTURE)
        register = next(r for r in records if "register user" in r["policy"])
        assert "user profile" in register["document"]

    def test_no_empty_document(self):
        records = parse_policies(_POLICY_FIXTURE)
        assert all(r["document"].strip() for r in records)

    def test_empty_log_returns_empty(self):
        assert parse_policies("") == []

    def test_missing_step5_marker_returns_empty(self):
        text = _POLICY_FIXTURE.replace(
            "[Strategic Design - Step5]: Create Policies Between Events And Commands", ""
        )
        assert parse_policies(text) == []
