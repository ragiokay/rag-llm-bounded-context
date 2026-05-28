# DDD Elements Log Loader
# Parses the debug log produced by the automated DDD extraction tool.
# Extracts three directly-mapped record types (no inference):
#   1. Domain Events  (Step 2 table) -> log_domain_events collection
#   2. Commands with Actors (Step 4 table) -> log_commands collection
#   3. Command-Event Pairs (Step 3 block) -> log_commands_events_pairs collection
#
# Example domain event record:
#   embed_text   : "The system verifies the user ."
#   domain_event : "user verified"
#
# Example command record:
#   embed_text   : "The user clicks INITIATE_MEETING button ."
#   command      : "initiate meeting"
#   user_roles   : "initiator"
#
# Example command-event pair record:
#   embed_text            : "The user clicks INITIATE_MEETING ... The initiated meeting is scheduled ."
#   domain_event          : "meeting scheduled"
#   command               : "initiate meeting"
#   commands_events_pairs : [["initiate meeting", "meeting scheduled"]]

import os
import re
import sys
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "")
COLLECTION_DOMAIN_EVENTS = f"{COLLECTION_PREFIX}log_domain_events"
COLLECTION_COMMANDS = f"{COLLECTION_PREFIX}log_commands"
COLLECTION_PAIRS = f"{COLLECTION_PREFIX}log_commands_events_pairs"
MODEL_NAME = "all-MiniLM-L6-v2"

# Matches the leading log prefix on timestamped lines, e.g.:
#   [2026-05-12 00:05:12][DEBUG]
_LOG_PREFIX = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\](?:\[DEBUG\])?\s*")


def _strip_prefix(line: str) -> str:
    return _LOG_PREFIX.sub("", line).strip()


def _parse_table(section: str) -> list[dict]:
    """
    Parse an ASCII pipe-delimited table from a log section.
    Returns a list of row dicts keyed by lowercased column headers.
    Skips separator lines (+---+) and blank lines.
    """
    headers: list[str] = []
    rows: list[dict] = []

    for raw in section.splitlines():
        line = _strip_prefix(raw).strip()
        if not line or line.startswith("+"):
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not headers:
            headers = [h.lower().replace(" ", "_") for h in cells]
        else:
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))

    return rows


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    """Return the substring of text between two markers."""
    s = text.find(start_marker)
    if s == -1:
        return ""
    e = text.find(end_marker, s + len(start_marker))
    return text[s:e] if e != -1 else text[s:]


def parse_domain_events(log_text: str) -> list[dict]:
    """
    Parse Step 2 Domain Events table.
    Columns: Domain Event | Aggregate State | UC ID | Sentence
    Returns list of payload dicts ready for Qdrant.
    """
    section = _section_between(
        log_text,
        "[Domain Events] ===",
        "[Strategic Design - Step3]",
    )
    rows = _parse_table(section)

    records = []
    for row in rows:
        event = row.get("domain_event", "").strip()
        sentence = row.get("sentence", "").strip()
        if not event or not sentence:
            continue
        records.append({
            "domain_event": event,
            "source_phrase": sentence,
            "document": sentence,
        })
    return records


def parse_commands(log_text: str) -> list[dict]:
    """
    Parse Step 4 Commands table (with actors assigned).
    Columns: Command | Actor | UC ID | Sentence
    Returns list of payload dicts ready for Qdrant.
    Actor values like 'name: user' are normalised to just 'user'.
    Rows with Actor == 'None' omit the user_roles field.
    """
    # Step 4 has its own [Commands] table; locate it after the Step4 header
    step4_start = log_text.find("[Strategic Design - Step4]: Assign Actors For Commands")
    if step4_start == -1:
        return []

    section = _section_between(
        log_text[step4_start:],
        "[Commands] ===",
        "[Strategic Design - Step5]",
    )
    rows = _parse_table(section)

    records = []
    for row in rows:
        command = row.get("command", "").strip()
        sentence = row.get("sentence", "").strip()
        actor_raw = row.get("actor", "").strip()
        if not command or not sentence:
            continue

        payload: dict = {
            "command": command,
            "source_phrase": sentence,
            "document": sentence,
        }

        if actor_raw and actor_raw.lower() != "none":
            actor = re.sub(r"^name:\s*", "", actor_raw, flags=re.IGNORECASE).strip()
            if actor:
                payload["user_roles"] = actor

        records.append(payload)
    return records


_PAIR_RE = re.compile(r'Event-Command found:\s*"([^"]+)"\s*->\s*"([^"]+)"')
_SENTENCE_RE = re.compile(r'sentence:\s*(.+)')


def parse_command_event_pairs(log_text: str) -> list[dict]:
    """
    Parse Step 3 Event-Command pairs.
    Format (per pair, 3 lines):
        [timestamp][DEBUG] Event-Command found: "event"->"command"
                           sentence: <compound sentence>
                           Causal prediction confidence: <float>
    Returns list of payload dicts ready for Qdrant.
    commands_events_pairs stores [[command, event]] as list-of-list.
    """
    section = _section_between(
        log_text,
        "[Strategic Design - Step3]: Pair Commands with Events.",
        "[Strategic Design - Step4]: Assign Actors For Commands",
    )
    if not section:
        return []

    records = []
    lines = section.splitlines()
    i = 0
    while i < len(lines):
        stripped = _strip_prefix(lines[i])
        m = _PAIR_RE.search(stripped)
        if m:
            event = m.group(1).strip()
            command = m.group(2).strip()
            if i + 1 < len(lines):
                next_stripped = _strip_prefix(lines[i + 1])
                sm = _SENTENCE_RE.match(next_stripped)
                if sm:
                    sentence = sm.group(1).strip()
                    if event and command and sentence:
                        records.append({
                            "domain_event": event,
                            "command": command,
                            "commands_events_pairs": [[command, event]],
                            "source_phrase": sentence,
                            "document": sentence,
                        })
                    i += 2
                    continue
        i += 1
    return records


def _make_id(prefix: str, payload: dict) -> str:
    key = f"{prefix}|{payload.get('domain_event', '')}|{payload.get('command', '')}|{payload.get('document', '')[:80]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def embed_and_store(
    records: list[dict],
    embed_key: str,
    collection: str,
    model: SentenceTransformer,
    id_prefix: str,
    batch_size: int = 256,
) -> int:
    """Embed records and upsert into Qdrant. Returns count written."""
    if not records:
        return 0

    client = QdrantClient(url=QDRANT_URL, timeout=300)
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        texts = [r[embed_key] for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        points = [
            PointStruct(
                id=_make_id(id_prefix, r),
                vector=embeddings[j],
                payload=r,
            )
            for j, r in enumerate(batch)
        ]
        client.upsert(collection_name=collection, points=points)
        written += len(batch)
        print(f"  [{written}/{len(records)}] written", end="\r")
    print()
    return written


def run(log_path: str | None = None):
    """
    Full pipeline: read log -> parse -> embed -> Qdrant.
    Defaults to data/log/ directory; processes all *.log files found.
    """
    print("=== Log DDD Seed Start ===")

    if log_path is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "log")
        log_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".log")
        ]
    else:
        log_files = [log_path]

    if not log_files:
        print("[WARN] No .log files found.")
        return

    all_events: list[dict] = []
    all_commands: list[dict] = []
    all_pairs: list[dict] = []

    for fpath in log_files:
        print(f"Reading {fpath} ...")
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        events = parse_domain_events(text)
        cmds = parse_commands(text)
        pairs = parse_command_event_pairs(text)
        print(f"  domain events: {len(events)}, commands: {len(cmds)}, pairs: {len(pairs)}")
        all_events.extend(events)
        all_commands.extend(cmds)
        all_pairs.extend(pairs)

    print(f"\nTotal domain events : {len(all_events)}")
    print(f"Total commands      : {len(all_commands)}")
    print(f"Total pairs         : {len(all_pairs)}")

    if all_events:
        print(f"\nExample domain event:")
        ex = all_events[0]
        print(f"  domain_event : {ex['domain_event']}")
        print(f"  embed_text   : {ex['document']}")

    if all_commands:
        print(f"\nExample command:")
        ex = all_commands[0]
        print(f"  command      : {ex['command']}")
        print(f"  user_roles   : {ex.get('user_roles', '(none)')}")
        print(f"  embed_text   : {ex['document']}")

    if all_pairs:
        print(f"\nExample pair:")
        ex = all_pairs[0]
        print(f"  domain_event          : {ex['domain_event']}")
        print(f"  command               : {ex['command']}")
        print(f"  commands_events_pairs : {ex['commands_events_pairs']}")
        print(f"  embed_text            : {ex['document']}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"\nEmbedding with {MODEL_NAME}...")

    written_events = embed_and_store(
        all_events, "document", COLLECTION_DOMAIN_EVENTS, model, "log_event"
    )
    written_cmds = embed_and_store(
        all_commands, "document", COLLECTION_COMMANDS, model, "log_cmd"
    )
    written_pairs = embed_and_store(
        all_pairs, "document", COLLECTION_PAIRS, model, "log_pair"
    )

    print(f"\n=== Done ===")
    print(f"  {written_events} vectors -> '{COLLECTION_DOMAIN_EVENTS}'")
    print(f"  {written_cmds} vectors -> '{COLLECTION_COMMANDS}'")
    print(f"  {written_pairs} vectors -> '{COLLECTION_PAIRS}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to a specific log file (default: all files in data/log/)")
    args = parser.parse_args()
    run(log_path=args.log)
