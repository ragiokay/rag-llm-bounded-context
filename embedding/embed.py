# BFR5: Organize and clean datasets
# BFR6: Transfer datasets into vector space
# BFR7: Review and clean vector space
# IIR3: Embedding Module and Vector Database

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from causal_transform import transform_batch, CausalRelationRecord

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def embed_records(collection_name: str, records: list[CausalRelationRecord]) -> int:
    """
    BFR6: Embeds CausalRelationRecord objects into a ChromaDB collection.
    Metadata stored is the full DDD schema so the Prompt Generator can use
    fields directly without re-parsing.
    Returns the number of vectors written.
    """
    if not records:
        print(f"[embed] No valid records for '{collection_name}', skipping")
        return 0

    collection = chroma_client.get_or_create_collection(name=collection_name)

    print(f"[embed] Embedding {len(records)} records into '{collection_name}'...")
    texts = [r.embed_text for r in records]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[r.id for r in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[r.to_chroma_metadata() for r in records],
    )
    print(f"[embed] Stored {len(records)} vectors into '{collection_name}'")
    return len(records)


def review_collection(collection_name: str) -> bool:
    """
    BFR7: Sanity-checks a ChromaDB collection after writing.
    Returns True if the collection passes all checks.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    count = collection.count()
    print(f"[review] '{collection_name}': {count} vectors")

    if count == 0:
        print(f"[review] WARNING: '{collection_name}' is empty")
        return False

    sample = collection.peek(limit=3)
    for i, doc in enumerate(sample["documents"]):
        emb = sample["embeddings"][i]
        if emb is None or len(emb) == 0:
            print(f"[review] ERROR: empty embedding at index {i}")
            return False
        if any(v != v for v in emb):  # NaN check
            print(f"[review] ERROR: NaN in embedding at index {i}")
            return False

    print(f"[review] '{collection_name}' passed ✓")
    return True


def load_bpc() -> tuple[list[dict], list[str]]:
    """Download BPC dataset from HuggingFace and return (rows, domains)."""
    print("Downloading BPC dataset from HuggingFace (ibm-research/BPC)...")
    dataset = load_dataset("ibm-research/BPC")
    df = dataset["train"].to_pandas()
    print(f"Downloaded {len(df)} rows, columns: {df.columns.tolist()}")

    # qid is the dataset's own stable identifier; rename to "id" for transform_row
    df = df.rename(columns={"qid": "id"})
    rows = df.to_dict("records")
    domains = sorted(df["domain"].unique().tolist())
    return rows, domains


def run():
    """Full pipeline: HuggingFace -> transform -> embed -> ChromaDB -> review."""
    print("=== Embedding Module Start ===")

    all_rows, domains = load_bpc()
    print(f"[run] Domains: {domains}")

    total_written = 0
    total_skipped = 0

    for domain in domains:
        domain_rows = [r for r in all_rows if r["domain"] == domain]
        records, skipped = transform_batch(domain_rows)
        total_skipped += skipped

        collection_name = f"bpc_{domain.replace(' ', '_').lower()}"
        written = embed_records(collection_name, records)
        total_written += written
        review_collection(collection_name)

    print(f"=== Embedding Module Complete: {total_written} written, {total_skipped} skipped ===")


if __name__ == "__main__":
    run()
