class BusinessLogicFilter:
    """filter Business Logic data across all BPC domains"""
    def __init__(self, retriever):
        self.retriever = retriever
        # 1. All BPC DB
        self.bpc_collections = [
            "bpc_finance",
            "bpc_education",
            "bpc_retail",
            "bpc_medical",
            "bpc_human_resources",
            "bpc_transportation",
            "bpc_insurance",
            "bpc_manufacturing",
            "bpc_logistics"
        ]

    def get_clean_examples(self, query_text: str, top_k: int = 3) -> list:
        all_raw_results = []
        
        # 2. Extensive search: Search the databases of every industry and find the top K results.
        for collection in self.bpc_collections:
            try:
                results = self.retriever.search(query_text, collection, top_k)
                all_raw_results.extend(results)
            except Exception as e:
                # Protective net: If a data table is deleted or the connection fails, a warning will pop up but the search for the next table will continue.
                print(f"[Warning] Failed to search in {collection}: {e}")
        
        # 3. Global sorting: Reorder all retrieved results from highest to lowest based on vector similarity (score).
        all_raw_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 4. Only keep the top K records with the highest overall scores.
        global_top_k_results = all_raw_results[:top_k]
        
        # 5. Maintain original logic, only extract the output field.
        clean_rag_examples = []
        for item in global_top_k_results:
            if "output" in item:
                clean_rag_examples.append(item["output"])
                
        print(f"[Filter] BusinessLogic successfully retrieved {len(clean_rag_examples)} pieces of cross-domain data")
        return clean_rag_examples