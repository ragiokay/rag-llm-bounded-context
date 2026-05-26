import json
import re

class ResponseParser:
    """Responsible for accurately extracting JSON from the raw text returned by the model."""
    
    def sanitize_json(self, raw_text: str) -> dict:
        extracted_jsons = []
        
        # ==========================================
        # Step 1: Collect all possible JSON (regardless of whether the format is correct or not, dig them out first).
        # ==========================================
        
        # Strategy A: Prioritize JSON within Markdown tags (LLM's preferred format)
        md_match = re.search(r'```json\s*\n(.*?)\n```', raw_text, re.DOTALL | re.IGNORECASE)
        if md_match:
            try:
                extracted_jsons.append(json.loads(md_match.group(1)))
            except json.JSONDecodeError:
                pass
        # Strategy B: A JSON Scanner
        # It splits the JSON into multiple independent blocks, unlike greedy Regex which bundles everything together.
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            match = raw_text.find('{', pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(raw_text[match:])
                extracted_jsons.append(result)
                # Parsing successful. Move the scanner to the end of this JSON and continue searching.
                pos = match + index
            except json.JSONDecodeError:
                # Parsing failed (it might just be a regular bracket), move one space forward and continue searching.
                pos = match + 1

        # ==========================================
        # Step 2: Standardized Filtering and Validation (Quality Control Department)
        # ==========================================
        
        # Define the core keywords allowed (including Commands and Actors that Sal will use in the future).
        valid_keys = ["BusinessLogic", "DomainEvents", "Commands", "Actors"]
        
        for json_obj in extracted_jsons:
            if isinstance(json_obj, dict):
                # If this dictionary contains any of the keys we allow, it is the correct answer
                if any(key in json_obj for key in valid_keys):
                    return json_obj
                    
        # No valid DDD JSON was found.
        print(f"[Parser Error] Analyze JSON failed or correct DDD schema not found.")
        return {"error": "Invalid JSON format"}