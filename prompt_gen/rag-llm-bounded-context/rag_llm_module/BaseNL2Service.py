from abc import ABC, abstractmethod
from rag_llm_module.LLMConnector import LLMConnector
from rag_llm_module.ResponseParser import ResponseParser
from prompt_generator.BasePromptGenerator import BasePromptGenerator

class BaseNL2Service(ABC):
    def __init__(self):
        self.llm = LLMConnector()
        self.parser = ResponseParser()

    @abstractmethod
    def get_generator(self) -> BasePromptGenerator:
        pass

    def execute(self, input_text: str, top_k_examples: list) -> dict:
        generator = self.get_generator()
        prompt = generator.generate(top_k_examples, input_text)
        
        # call the llm
        raw_response = self.llm.post_to_llm(prompt)
        
        # Force print the original LLM response (use repr to display hidden characters)
        print("\n" + "="*40)
        print(f"[LLM Raw Response]:\n{repr(raw_response)}")
        print("="*40 + "\n")
        
        return self.parser.sanitize_json(raw_response)