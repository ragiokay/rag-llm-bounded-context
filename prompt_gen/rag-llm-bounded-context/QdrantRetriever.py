import os
import sys

# Environmental variables and path setting
os.environ.setdefault("QDRANT_URL", "http://140.112.90.146:6333")
os.environ.setdefault("COLLECTION_PREFIX", "spring2026SE_g1_rag_")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "embedding"))

from retrieve import query_similar

class QdrantRetriever:
    """database retriever, containing no business cleansing logic."""
    def __init__(self):
        self.prefix = os.environ.get("COLLECTION_PREFIX", "spring2026SE_g1_rag_")
        print(f"[Retriever] Initialized, target server: {os.environ.get('QDRANT_URL')}")

    def search(self, query_text: str, collection_suffix: str, top_k: int = 3) -> list:
        # Automatically assemble the complete Collection name
        full_collection_name = f"{self.prefix}{collection_suffix}"
        print(f"[Retriever] Searching for data tables: {full_collection_name}...")
        
        try:
            # call 睿傑 API
            results = query_similar(
                query_text=query_text, 
                collection=full_collection_name, 
                n_results=top_k
            )
            print(f"[Debug] Raw results from query_similar: {results}")
            return results
        except Exception as e:
            print(f"[Retriever] Query failed: {e}")
            return []