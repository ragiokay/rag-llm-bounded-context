from abc import ABC, abstractmethod
from prompt_generator.PromptBuilder import PromptBuilder

class BasePromptGenerator(ABC):
    """Template Method Pattern: Defines the standard assembly skeleton for all Prompt Generators."""
    
    def get_role(self) -> str:
        return "You are a Domain-Driven Design (DDD) expert."

    @abstractmethod
    def get_task(self) -> str: pass

    @abstractmethod
    def get_rules(self) -> str: pass

    @abstractmethod
    def get_output_schema(self) -> dict: pass

    def generate(self, top_k_data: list, target_text: str) -> str:
        builder = PromptBuilder()
        return (builder
            .add_role_and_task(self.get_role(), self.get_task())
            .add_rules(self.get_rules())
            .add_output_format(self.get_output_schema())
            .add_rag_examples(top_k_data)
            .add_user_query(target_text)
            .build())