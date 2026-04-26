# BFR5: Organize and clean datasets
# BFR6: Transfer datasets into vector space
# BFR7: Review and clean vector space
# IIR3: Embedding Module and Vector Database

import chromadb
from sentence_transformers import SentenceTransformer
from fetch_from_db import fetch_all, fetch_distinct_domains, fetch_by_domain

# Initialize embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Initialize ChromaDB (local persistent storage)
chroma_client = chromadb.PersistentClient(path="../vector_db")

def clean_record(row):
    """
    BFR5: Organize and clean a single record.
    Returns None if record is invalid.
    """
    phrase = str(row.get("phrase", "")).strip()
    question = str(row.get("question", "")).strip()

    if not phrase or not question:
        return None

    # Combine phrase + question as the text to embed
    text = f"{phrase} {question}"
    return text

def embed_to_collection(collection_name, rows):
    """
    BFR6: Convert cleaned records into vector space.
    Stores into a ChromaDB collection.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    texts = []
    ids = []
    metadatas = []

    for row in rows:
        text = clean_record(row)
        if text is None:
            print(f"[embed] Skipping invalid record id={row['id']}")
            continue

        texts.append(text)
        ids.append(str(row["id"]))
        metadatas.append({
            "answer": str(row["answer"]),
            "category": str(row["category"]),
            "domain": str(row["domain"])
        })

    if not texts:
        print(f"[embed] No valid records to embed for collection '{collection_name}'")
        return

    print(f"[embed] Embedding {len(texts)} records into '{collection_name}'...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"[embed] Stored {len(texts)} vectors into '{collection_name}'")

def review_collection(collection_name):
    """
    BFR7: Review and clean vector space.
    Checks completeness and consistency.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    count = collection.count()
    print(f"[review] Collection '{collection_name}': {count} vectors")

    # Check for empty collection
    if count == 0:
        print(f"[review] WARNING: Collection '{collection_name}' is empty!")
        return False

    # Sample check: peek at first 3 entries
    sample = collection.peek(limit=3)
    for i, doc in enumerate(sample["documents"]):
        embedding = sample["embeddings"][i]
        if embedding is None or len(embedding) == 0:
            print(f"[review] ERROR: Found empty embedding at index {i}")
            return False
        if any(v != v for v in list(embedding)):  # NaN check
            print(f"[review] ERROR: Found NaN in embedding at index {i}")
            return False

    print(f"[review] Collection '{collection_name}' passed review ✓")
    return True

def run():
    """
    Full pipeline:
    MySQL -> clean -> embed -> ChromaDB -> review
    """
    print("=== Embedding Module Start ===")

    # Fetch all data from MySQL
    all_rows = fetch_all()

    # Get distinct domains -> one collection per domain
    domains = fetch_distinct_domains()
    print(f"[run] Domains found: {domains}")

    for domain in domains:
        domain_rows = [r for r in all_rows if r["domain"] == domain]
        collection_name = f"bpc_{domain.replace(' ', '_').lower()}"
        embed_to_collection(collection_name, domain_rows)
        review_collection(collection_name)

    print("=== Embedding Module Complete ===")

if __name__ == "__main__":
    run()