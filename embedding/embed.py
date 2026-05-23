# BFR5: Organize and clean datasets
# BFR6: Transfer datasets into vector space
# BFR7: Review and clean vector space
# IIR3: Embedding Module and Vector Database

import os
import sys
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from causal_transform import transform_batch, CausalRelationRecord

MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "")

model = SentenceTransformer(MODEL_NAME)

_qdrant_client = None


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=300)
    return _qdrant_client


def embed_records(collection_name: str, records: list[CausalRelationRecord],
                  batch_size: int = 128) -> int:
    """
    BFR6: Embeds CausalRelationRecord objects into a Qdrant collection.
    Metadata stored is the full DDD schema so the Prompt Generator can use
    fields directly without re-parsing.
    Returns the number of vectors written.
    """
    if not records:
        print(f"[embed] No valid records for '{collection_name}', skipping")
        return 0

    client = _get_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    print(f"[embed] Embedding {len(records)} records into '{collection_name}'...")
    texts = [r.embed_text for r in records]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    written = 0
    for i in range(0, len(records), batch_size):
        batch_records = records[i: i + batch_size]
        batch_embeddings = embeddings[i: i + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, r.id)),
                vector=batch_embeddings[j],
                payload={**r.to_chroma_metadata(), "document": r.embed_text},
            )
            for j, r in enumerate(batch_records)
        ]
        client.upsert(collection_name=collection_name, points=points)
        written += len(batch_records)
        print(f"  [{written}/{len(records)}] upserted", end="\r")
    print()
    print(f"[embed] Stored {len(records)} vectors into '{collection_name}'")
    return len(records)


def review_collection(collection_name: str) -> bool:
    """
    BFR7: Sanity-checks a Qdrant collection after writing.
    Returns True if the collection passes all checks.
    """
    client = _get_client()
    count = client.get_collection(collection_name).points_count
    print(f"[review] '{collection_name}': {count} vectors")

    if count == 0:
        print(f"[review] WARNING: '{collection_name}' is empty")
        return False

    points, _ = client.scroll(
        collection_name=collection_name, limit=3, with_vectors=True, with_payload=True
    )
    for i, point in enumerate(points):
        emb = point.vector
        if emb is None or len(emb) == 0:
            print(f"[review] ERROR: empty embedding at index {i}")
            return False
        if any(v != v for v in emb):  # NaN check
            print(f"[review] ERROR: NaN in embedding at index {i}")
            return False

    print(f"[review] '{collection_name}' passed OK")
    return True


def load_bpc() -> tuple[list[dict], list[str]]:
    """Download BPC dataset from HuggingFace and return (rows, domains)."""
    print("Downloading BPC dataset from HuggingFace (ibm-research/BPC)...")
    dataset = load_dataset("ibm-research/BPC")
    df = dataset["train"].to_pandas()
    print(f"Downloaded {len(df)} rows, columns: {df.columns.tolist()}")

    # Use the dataframe index as id — qid is not unique across domains
    df["id"] = df.index.astype(str)
    rows = df.to_dict("records")
    domains = sorted(df["domain"].unique().tolist())
    return rows, domains


def run():
    """Full pipeline: HuggingFace -> transform -> embed -> Qdrant -> review."""
    print("=== Embedding Module Start ===")

    all_rows, domains = load_bpc()
    print(f"[run] Domains: {domains}")

    total_written = 0
    total_skipped = 0

    for domain in domains:
        domain_rows = [r for r in all_rows if r["domain"] == domain]
        records, skipped = transform_batch(domain_rows)
        total_skipped += skipped

        collection_name = f"{COLLECTION_PREFIX}bpc_{domain.replace(' ', '_').lower()}"
        written = embed_records(collection_name, records)
        total_written += written
        review_collection(collection_name)

    print(f"=== Embedding Module Complete: {total_written} written, {total_skipped} skipped ===")


if __name__ == "__main__":
    run()
