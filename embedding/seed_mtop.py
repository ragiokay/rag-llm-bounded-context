# MTOP Dataset Loader
# Filters command-type intents from MTOP and embeds them into ChromaDB as
# Event Storming Command examples for the Prompt Generator.
#
# Only command-like intents (CREATE_*, DELETE_*, SEND_*, etc.) are stored.
# Query intents (GET_*, QUESTION_*, IS_TRUE_*) are discarded at parse time.
#
# Example record:
#   text        : "Remind me to start cooking dinner in 10 minutes"
#   intent      : CREATE_REMINDER
#   command     : Create Reminder
#   trigger_span: "Remind me"
#   policy      : "command: CREATE_REMINDER"

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer
from causal_transform import CausalRelationRecord

CHROMA_PATH = "../vector_db"
COLLECTION_NAME = "mtop_commands"
MODEL_NAME = "all-MiniLM-L6-v2"

# Intent prefix → classification
_QUERY_PREFIXES = ("GET_", "QUESTION_", "IS_TRUE_",)
_COMMAND_PREFIXES = (
    "CREATE_", "DELETE_", "UPDATE_", "SET_", "SEND_", "ADD_", "REMOVE_",
    "SHARE_", "ANSWER_", "END_", "IGNORE_", "SNOOZE_", "SILENCE_",
    "START_", "STOP_", "PAUSE_", "RESUME_", "SWITCH_", "SUBTRACT_",
    "PLAY_",
)

# Words to skip at the start of an utterance before the trigger verb
_LEAD_SKIP = frozenset({"please", "can", "could", "would", "i'd", "i", "you"})
# Words that end the trigger span
_SPAN_STOP = frozenset({"for", "at", "on", "by", "in", "to", "that", "when",
                         "if", "until", "about", "and", "or"})


def classify_mtop_intent(intent: str) -> str:
    """
    Returns "command", "query", or "review" for a given MTOP intent label.
    Handles both "IN:CREATE_REMINDER" and "CREATE_REMINDER" formats.
    """
    name = intent.removeprefix("IN:")
    if name.startswith(_QUERY_PREFIXES):
        return "query"
    if name.startswith(_COMMAND_PREFIXES):
        return "command"
    return "review"


def intent_to_command_name(intent: str) -> str:
    """Convert MTOP intent to a DDD-style command name.
    E.g. CREATE_REMINDER -> "Create Reminder", SEND_MESSAGE -> "Send Message"
    """
    name = intent.removeprefix("IN:")
    parts = name.split("_", 1)
    if len(parts) == 2:
        action, obj = parts
        return f"{action.capitalize()} {obj.replace('_', ' ').title()}"
    return name.replace("_", " ").title()


def intent_to_aggregate(intent: str) -> str:
    """Extract the object/domain part of the intent as the aggregate.
    E.g. CREATE_REMINDER -> "REMINDER", SEND_MESSAGE -> "MESSAGE"
    """
    name = intent.removeprefix("IN:")
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else name


def extract_trigger_span(text: str) -> str:
    """
    Extract the imperative command trigger span from an utterance.

    Heuristic: skip leading courtesy/modal words, then take words until
    a preposition or subordinator is hit (capped at 4 words).

    "Remind me to start cooking" -> "Remind me"
    "Set an alarm for Thursday"  -> "Set an alarm"
    "Please send a message to Bob" -> "send a message"
    """
    tokens = text.strip().split()
    if not tokens:
        return ""

    # Skip leading courtesy words
    start = 0
    while start < len(tokens) and tokens[start].lower().rstrip(",'") in _LEAD_SKIP:
        start += 1

    end = start
    for i in range(start, min(start + 4, len(tokens))):
        word = tokens[i].lower().rstrip(".,?!")
        if i > start and word in _SPAN_STOP:
            break
        end = i + 1

    if end <= start:
        end = min(start + 1, len(tokens))

    return " ".join(tokens[start:end])


def parse_record(row: dict, idx: int) -> "CausalRelationRecord | None":
    """
    Parse one MTOP row into a CausalRelationRecord.
    Returns None for query-type intents and rows missing required fields.

    Supports both mteb/mtop_intent format (label_text) and raw MTOP format (intent).
    """
    text = str(row.get("text") or row.get("utterance") or "").strip()
    intent = str(row.get("label_text") or row.get("intent") or "").strip()

    if not text or not intent:
        return None
    if classify_mtop_intent(intent) != "command":
        return None

    command_name = intent_to_command_name(intent)
    aggregate = intent_to_aggregate(intent)
    trigger = extract_trigger_span(text)
    clean_intent = intent.removeprefix("IN:")

    try:
        return CausalRelationRecord(
            id=f"mtop_{idx}",
            domain_event=text,
            command=command_name,
            policy=f"command: {clean_intent}",
            aggregate=aggregate,
            bounded_context="MTOP",
            source_phrase=text,
            embed_text=text,
            trigger_span=trigger or None,
        )
    except Exception:
        return None


def embed_and_store(records: list[CausalRelationRecord],
                    collection, model: SentenceTransformer,
                    batch_size: int = 256) -> int:
    """Embeds records in batches and upserts into ChromaDB. Returns count written."""
    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        texts = [r.embed_text for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[r.id for r in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[r.to_chroma_metadata() for r in batch],
        )
        written += len(batch)
        print(f"  [{written}/{len(records)}] written", end="\r")
    print()
    return written


def run(limit: int | None = None, lang: str = "en"):
    """
    Full pipeline: HuggingFace download -> filter -> embed -> ChromaDB.

    Args:
        limit: cap on number of rows to process (None = all).
               Use a small number (e.g. 200) for a quick test run.
        lang:  MTOP language config (default "en").
    """
    print("=== MTOP Seed Start ===")

    try:
        from datasets import load_dataset
        print(f"Loading MTOP (mteb/mtop_intent, config={lang}) from HuggingFace...")
        dataset = load_dataset("mteb/mtop_intent", lang, trust_remote_code=True)
        rows = list(dataset["train"])
        if "validation" in dataset:
            rows += list(dataset["validation"])
        if "test" in dataset:
            rows += list(dataset["test"])
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        raise

    if limit:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} rows")

    print("Parsing command-type intents (filtering out query/review)...")
    all_records: list[CausalRelationRecord] = []
    skipped = 0
    for idx, row in enumerate(rows):
        record = parse_record(row, idx)
        if record is None:
            skipped += 1
        else:
            all_records.append(record)

    print(f"Parsed {len(all_records)} command records ({skipped} query/review skipped)")

    if not all_records:
        print("[WARN] No records parsed. Check dataset format.")
        return

    # Show example
    ex = all_records[0]
    print("\n--- Example record ---")
    print(f"  Text         : {ex.embed_text}")
    print(f"  Command      : {ex.command}")
    print(f"  Trigger span : {ex.trigger_span}")
    print(f"  Policy       : {ex.policy}")
    print(f"  Aggregate    : {ex.aggregate}")
    print("----------------------\n")

    print(f"Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    written = embed_and_store(all_records, collection, model)
    print(f"=== Done: {written} vectors in collection '{COLLECTION_NAME}' ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows (e.g. --limit 200 for quick test)")
    parser.add_argument("--lang", type=str, default="en",
                        help="MTOP language config (default: en)")
    args = parser.parse_args()
    run(limit=args.limit, lang=args.lang)
