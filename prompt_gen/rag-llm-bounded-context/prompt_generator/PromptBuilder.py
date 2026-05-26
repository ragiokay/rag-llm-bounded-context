import json

class PromptBuilder:
    """It is responsible for assembling the scattered text into the XML-tag Prompt format specific to LLaMA 3.2."""
    def __init__(self):
        self.prompt = ""
        self.MAX_RAG_CHARS = 2000 * 4

    def add_role_and_task(self, role: str, task: str):
        self.prompt += f"<|start_header_id|>system<|end_header_id|>\n[ROLE]\n{role}\n[TASK]\n{task}\n"
        return self

    def add_rules(self, rules: str):
        self.prompt += f"\n[RULES]\n{rules}\n"
        return self

    def add_output_format(self, schema: dict):
        self.prompt += f"\n[OUTPUT FORMAT]\nYou MUST output ONLY valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n"
        return self

    def add_rag_examples(self, examples: list):
        if not examples:
            return self
            
        self.prompt += "\n[RAG EXAMPLES (Learn from these)]\n"
        
        current_chars = 0
        accepted_count = 0
        
        for i, example in enumerate(examples):
            # Convert the case into a string
            example_str = json.dumps(example, indent=2, ensure_ascii=False)
            
            # IIR2: verify length
            if current_chars + len(example_str) > self.MAX_RAG_CHARS:
                print(f"[PromptBuilder] Token capacity has reached its limit! RAG cases after the {i+1}th transaction have been discarded")
                break
                
            self.prompt += f"Example {i+1}:\n{example_str}\n\n"
            current_chars += len(example_str)
            accepted_count += 1
            
        print(f"[PromptBuilder] successfully loaded {accepted_count} pieces of RAG cases ( {current_chars} chars in total).")
        return self

    def add_user_query(self, query: str):
        self.prompt += f"<|start_header_id|>user<|end_header_id|>\n[TARGET BUSINESS LOGIC]\n{query}\n<|start_header_id|>assistant<|end_header_id|>\n"
        return self

    def build(self) -> str:
        return self.prompt