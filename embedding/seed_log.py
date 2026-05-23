# DDD Elements Log Loader
# Parses the debug log produced by the automated DDD extraction tool.
# Extracts two directly-mapped record types (no inference):
#   1. Domain Events  (Step 2 table) -> log_domain_events collection
#   2. Commands with Actors (Step 4 table) -> log_commands collection
#
# Example domain event record:
#   embed_text   : "The system verifies the user ."
#   domain_event : "user verified"
#
# Example command record:
#   embed_text   : "The user clicks INITIATE_MEETING button ."
#   command      : "initiate meeting"
#   user_roles   : "initiator"

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


def _make_id(prefix: str, payload: dict) -> str:
    key = f"{prefix}|{payload.get('domain_event') or payload.get('command', '')}|{payload.get('document', '')[:80]}"
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

    for fpath in log_files:
        print(f"Reading {fpath} ...")
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        events = parse_domain_events(text)
        cmds = parse_commands(text)
        print(f"  domain events: {len(events)}, commands: {len(cmds)}")
        all_events.extend(events)
        all_commands.extend(cmds)

    print(f"\nTotal domain events : {len(all_events)}")
    print(f"Total commands      : {len(all_commands)}")

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

    model = SentenceTransformer(MODEL_NAME)
    print(f"\nEmbedding with {MODEL_NAME}...")

    written_events = embed_and_store(
        all_events, "document", COLLECTION_DOMAIN_EVENTS, model, "log_event"
    )
    written_cmds = embed_and_store(
        all_commands, "document", COLLECTION_COMMANDS, model, "log_cmd"
    )

    print(f"\n=== Done ===")
    print(f"  {written_events} vectors -> '{COLLECTION_DOMAIN_EVENTS}'")
    print(f"  {written_cmds} vectors -> '{COLLECTION_COMMANDS}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to a specific log file (default: all files in data/log/)")
    args = parser.parse_args()
    run(log_path=args.log)
