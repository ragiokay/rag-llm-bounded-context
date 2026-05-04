# MAVEN-ERE Dataset Loader
# Parses CAUSE/PRECONDITION relations into CausalRelationRecord and embeds into ChromaDB.
#
# Input  (what gets embedded): the sentence containing the cause event trigger
# Output (ChromaDB metadata) : cause trigger word + effect trigger word + relation type
#
# Example:
#   Input : "The match was postponed because of a thunderstorm."
#   Output: cause="thunderstorm", effect="match was postponed", policy="CAUSE: thunderstorm → postponed"

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer
from causal_transform import CausalRelationRecord

CHROMA_PATH = "../vector_db"
COLLECTION_NAME = "maven_ere_causal"
MODEL_NAME = "all-MiniLM-L6-v2"
RELATION_TYPES = {"CAUSE", "PRECONDITION"}


def _sentence(doc: dict, sent_id: int) -> str:
    """Return sentence as string; handles both str and tokenized list formats."""
    raw = doc["sentences"][sent_id]
    if isinstance(raw, list):
        return " ".join(raw)
    return str(raw)


def parse_document(doc: dict) -> list[CausalRelationRecord]:
    """
    Parse one MAVEN-ERE document into CausalRelationRecord list.

    Official THU-KEG format:
        doc["sentences"]          — list of sentences
        doc["events"]             — list of events; each event has "id", "type", "mentions"
        doc["causal_relations"]   — dict: {"CAUSE": [[eid1, eid2], ...], "PRECONDITION": [...]}

    event map is keyed by EVENT id (not mention id).
    embed_text = sentence containing the cause event trigger.
    """
    # Build event lookup: event_id -> {trigger, sentence, event_type}
    event_map: dict = {}
    for event in doc.get("events", []):
        event_type = event.get("type", "unknown")
        # Use the first mention as the representative trigger
        mentions = event.get("mentions", [])
        if not mentions:
            continue
        mention = mentions[0]
        sent_id = mention.get("sent_id", 0)
        event_map[event["id"]] = {
            "trigger": mention.get("trigger_word", ""),
            "sentence": _sentence(doc, sent_id),
            "event_type": event_type,
        }

    doc_title = doc.get("title", "unknown")
    doc_id = str(doc.get("id", ""))
    records: list[CausalRelationRecord] = []

    # causal_relations is a dict: {relation_type: [[head_id, tail_id], ...]}
    causal_relations = doc.get("causal_relations", {})
    if not isinstance(causal_relations, dict):
        return records

    for relation, pairs in causal_relations.items():
        if relation not in RELATION_TYPES:
            continue
        for idx, pair in enumerate(pairs):
            if len(pair) < 2:
                continue
            head = event_map.get(pair[0])
            tail = event_map.get(pair[1])
            if not head or not tail:
                continue

            policy = f"{relation}: {head['trigger']} → {tail['trigger']}"
            try:
                record = CausalRelationRecord(
                    id=f"maven_{doc_id}_{relation}_{idx}",
                    domain_event=head["sentence"],
                    command=tail["sentence"],
                    policy=policy,
                    aggregate=head["event_type"],
                    bounded_context=doc_title,
                    source_phrase=head["sentence"],
                    embed_text=head["sentence"],
                )
                records.append(record)
            except Exception:
                continue

    return records


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


def run(limit_docs: int | None = None):
    """
    Full pipeline: HuggingFace download -> parse -> embed -> ChromaDB.

    Args:
        limit_docs: cap on number of documents to process (None = all).
                    Use a small number (e.g. 100) for a quick test run.
    """
    print("=== MAVEN-ERE Seed Start ===")

    # --- Load dataset ---
    # Priority 1: local JSONL file (official THU-KEG download)
    # Priority 2: HuggingFace unofficial mirror (fallback)
    default_jsonl = os.path.join(os.path.dirname(__file__), "..", "data", "maven_ere", "train.jsonl")
    jsonl_path = os.environ.get("MAVEN_ERE_PATH", default_jsonl)

    if os.path.exists(jsonl_path):
        print(f"Loading MAVEN-ERE from local file: {jsonl_path}")
        import json
        with open(jsonl_path, "r", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[INFO] Local file not found at {jsonl_path}")
        print("Falling back to HuggingFace (Nofing/maven-ere-json)...")
        print("Tip: Download official data from https://github.com/THU-KEG/MAVEN-ERE")
        print("     and place train.jsonl at data/maven_ere/train.jsonl for better reliability.")
        try:
            from datasets import load_dataset
            dataset = load_dataset("Nofing/maven-ere-json", split="train", trust_remote_code=True)
            docs = list(dataset)
        except Exception as e:
            print(f"[ERROR] Could not load dataset: {e}")
            raise
    if limit_docs:
        docs = docs[:limit_docs]
    print(f"Loaded {len(docs)} documents")

    print("Parsing cause-effect relations...")
    all_records: list[CausalRelationRecord] = []
    for doc in docs:
        all_records.extend(parse_document(doc))
    print(f"Parsed {len(all_records)} CAUSE/PRECONDITION pairs")

    if not all_records:
        print("[WARN] No records parsed. Check dataset format.")
        return

    # Show example
    ex = all_records[0]
    print("\n--- Example pair ---")
    print(f"  Input  (embed): {ex.embed_text[:120]}")
    print(f"  Cause trigger : {ex.policy.split(': ')[1].split(' → ')[0]}")
    print(f"  Effect trigger: {ex.policy.split(' → ')[1]}")
    print(f"  Policy        : {ex.policy}")
    print(f"  Context       : {ex.bounded_context}")
    print("--------------------\n")

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
                        help="Limit number of documents (e.g. --limit 50 for quick test)")
    args = parser.parse_args()
    run(limit_docs=args.limit)
