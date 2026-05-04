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

    MAVEN-ERE structure used:
        doc["sentences"]        — list of sentences
        doc["events"]           — list of events with mentions (trigger_word, sent_id)
        doc["causal_relations"] — list of {head_id, tail_id, relation}
    """
    # Build event mention lookup: mention_id -> {trigger, sentence, event_type}
    event_map: dict = {}
    for event in doc.get("events", []):
        event_type = event.get("type", "unknown")
        for mention in event.get("mentions", []):
            sent_id = mention.get("sent_id", 0)
            event_map[mention["id"]] = {
                "trigger": mention.get("trigger_word", ""),
                "sentence": _sentence(doc, sent_id),
                "event_type": event_type,
            }

    doc_title = doc.get("title", "unknown")
    doc_id = str(doc.get("id", ""))
    records: list[CausalRelationRecord] = []

    for idx, rel in enumerate(doc.get("causal_relations", [])):
        if rel.get("relation") not in RELATION_TYPES:
            continue

        head = event_map.get(rel.get("head_id", ""))
        tail = event_map.get(rel.get("tail_id", ""))
        if not head or not tail:
            continue

        cause_trigger = head["trigger"]
        effect_trigger = tail["trigger"]
        relation = rel.get("relation", "CAUSE")

        # embed_text = the sentence containing the cause (what we search with)
        embed_text = head["sentence"]
        # command = the sentence containing the effect (context of consequence)
        command = tail["sentence"]
        policy = f"{relation}: {cause_trigger} → {effect_trigger}"

        try:
            record = CausalRelationRecord(
                id=f"maven_{doc_id}_{idx}",
                domain_event=head["sentence"],
                command=command,
                policy=policy,
                aggregate=head["event_type"],
                bounded_context=doc_title,
                source_phrase=head["sentence"],
                embed_text=embed_text,
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

    print("Loading MAVEN-ERE from HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("Nofing/maven-ere-json", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        print("Try: pip install datasets")
        raise

    docs = list(dataset)
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
