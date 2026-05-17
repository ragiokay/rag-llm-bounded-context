# MTOP Dataset Loader
# Filters command-type intents from MTOP and embeds them into Qdrant as
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
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from causal_transform import CausalRelationRecord

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
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


def load_mtop_tsv(path: str) -> list[dict]:
    """
    Load the official Facebook MTOP raw TSV file.

    Tab-separated columns (no header):
        0: id  1: intent  2: slots  3: utterance  4: domain  5: locale  6: parse  7: tokenSpans

    Returns list of dicts with keys: intent, text, domain.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append({
                "intent": parts[1],
                "text":   parts[3],
                "domain": parts[4] if len(parts) > 4 else "mtop",
            })
    return rows


def parse_record(row: dict, idx: int) -> "CausalRelationRecord | None":
    """
    Parse one MTOP row into a CausalRelationRecord.
    Returns None for query-type intents and rows missing required fields.

    Accepts raw TSV rows (keys: intent, text, domain) and
    HF mteb/mtop_intent rows (keys: label_text, text).
    """
    text   = str(row.get("text") or row.get("utterance") or "").strip()
    intent = str(row.get("intent") or row.get("label_text") or "").strip()
    domain = str(row.get("domain") or "MTOP").strip()

    if not text or not intent:
        return None
    if classify_mtop_intent(intent) != "command":
        return None

    command_name = intent_to_command_name(intent)
    aggregate = intent_to_aggregate(intent).lower()
    trigger = extract_trigger_span(text)
    clean_intent = intent.removeprefix("IN:").lower()

    try:
        return CausalRelationRecord(
            id=f"mtop_{idx}",
            domain_event=text,
            command=command_name,
            policy=f"command: {clean_intent}",
            aggregate=aggregate,
            bounded_context=domain,
            source_phrase=text,
            embed_text=text,
            trigger_span=trigger or None,
        )
    except Exception:
        return None


def embed_and_store(records: list[CausalRelationRecord],
                    collection: str, model: SentenceTransformer,
                    batch_size: int = 256) -> int:
    """Embeds records in batches and upserts into Qdrant. Returns count written."""
    if not records:
        return 0

    client = QdrantClient(url=QDRANT_URL)

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        texts = [r.embed_text for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, r.id)),
                vector=embeddings[j],
                payload={**r.to_chroma_metadata(), "document": r.embed_text},
            )
            for j, r in enumerate(batch)
        ]
        client.upsert(collection_name=collection, points=points)
        written += len(batch)
        print(f"  [{written}/{len(records)}] written", end="\r")
    print()
    return written


def run(limit: int | None = None, lang: str = "en"):
    """
    Full pipeline: local TSV files -> filter -> embed -> Qdrant.

    Expects the official Facebook MTOP release laid out as:
        data/mtop/<lang>/train.txt   (required)
        data/mtop/<lang>/eval.txt    (optional)
        data/mtop/<lang>/test.txt    (optional)

    Override the base directory with the MTOP_PATH environment variable.

    Args:
        limit: cap on number of rows to process (None = all).
               Use a small number (e.g. 200) for a quick test run.
        lang:  language subfolder (default "en").
    """
    print("=== MTOP Seed Start ===")

    base = os.environ.get(
        "MTOP_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "mtop"),
    )
    lang_dir = os.path.join(base, lang)

    rows: list[dict] = []
    for split in ("train.txt", "eval.txt", "test.txt"):
        fpath = os.path.join(lang_dir, split)
        if os.path.exists(fpath):
            split_rows = load_mtop_tsv(fpath)
            rows.extend(split_rows)
            print(f"  Loaded {len(split_rows):>6} rows from {fpath}")
        else:
            print(f"  [skip] {fpath} not found")

    if not rows:
        print(
            f"\n[ERROR] No MTOP files found under: {lang_dir}\n"
            f"  Download the dataset from https://fb.me/mtop_dataset\n"
            f"  and place the .txt files at:  data/mtop/{lang}/train.txt\n"
            f"  or set the MTOP_PATH env var to your base directory.\n"
        )
        raise FileNotFoundError(f"No MTOP files found in {lang_dir}")

    if limit:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} rows total")

    print("Parsing command-type intents (filtering out query/review)...")
    all_records: list[CausalRelationRecord] = []
    skipped_rows: list[dict] = []
    for idx, row in enumerate(rows):
        record = parse_record(row, idx)
        if record is None:
            skipped_rows.append(row)
        else:
            all_records.append(record)

    print(f"[filter] kept: {len(all_records)}, skipped: {len(skipped_rows)}")

    # Save filter audit files so you can inspect what was kept vs dropped
    kept_path    = os.path.join(lang_dir, "filtered_kept.txt")
    skipped_path = os.path.join(lang_dir, "filtered_skipped.txt")
    os.makedirs(lang_dir, exist_ok=True)

    with open(kept_path, "w", encoding="utf-8") as f:
        for r in all_records:
            intent = r.policy.replace("command: ", "").upper()
            f.write(f"{intent}\t{r.source_phrase}\n")

    with open(skipped_path, "w", encoding="utf-8") as f:
        for row in skipped_rows:
            intent = str(row.get("intent") or row.get("label_text") or "")
            text   = str(row.get("text") or row.get("utterance") or "")
            f.write(f"{intent}\t{text}\n")

    print(f"Saved kept    → {kept_path}")
    print(f"Saved skipped → {skipped_path}")

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

    written = embed_and_store(all_records, COLLECTION_NAME, model)
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
