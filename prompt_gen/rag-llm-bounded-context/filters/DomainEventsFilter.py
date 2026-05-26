class DomainEventsFilter:
    """filter Domain Events data"""
    def __init__(self, retriever):
        self.retriever = retriever # inject Retriever
        self.target_collection = "log_domain_events"

    def get_clean_examples(self, query_text: str, top_k: int = 3) -> list:
        # 1. retrieve the original data
        raw_results = self.retriever.search(query_text, self.target_collection, top_k)
        
        # 2.append the filter data
        clean_rag_examples = []
        for item in raw_results:
            if "output" in item:
                clean_rag_examples.append(item["output"])
                
        print(f"[Filter] DomainEvents successfully filter {len(clean_rag_examples)} pieces of data.")
        return clean_rag_examples