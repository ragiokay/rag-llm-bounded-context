import requests
import json
import re

class LLMConnector:
    """lab server vllm + llama 3.2 1b"""
    def __init__(self, endpoint="http://140.112.90.146:8088/v1/completions"):
        self.endpoint = endpoint
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"

    def post_to_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.1,
            "stop": ["<|eot_id|>"]
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            print(f"[vLLM Error] Connection failed: {e}")
            return "{}"

# class ResponseParser:
#     """Responsible for accurately extracting curly braces JSON from the raw text returned by the model."""
#     def sanitize_json(self, raw_text: str) -> dict:
#         try:
#             json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group(0))
#             return {}
#         except json.JSONDecodeError:
#             print(f"[Parser Error] Analyze JSON failed: {raw_text}")
#             return {"error": "Invalid JSON format"}