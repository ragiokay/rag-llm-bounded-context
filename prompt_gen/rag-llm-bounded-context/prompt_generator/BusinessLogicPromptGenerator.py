from prompt_generator.BasePromptGenerator import BasePromptGenerator

class BusinessLogicPromptGenerator(BasePromptGenerator):
    def get_task(self) -> str:
        return "Analyze the raw user input and extract the core Business Logic, concepts, and triggering conditions clearly and concisely."

    def get_rules(self) -> str:
        return (
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY a valid JSON object. No conversational text, no preambles.\n"
            "2. Represent all logic, conditions, and initial triggers as short key terms or noun phrases (e.g., 'Cart submission', 'VIP 10% discount').\n"
            "3. Do not use full sentences. Break down complex rules into discrete concepts."
        )

    def get_output_schema(self) -> dict:
        return {
            "BusinessLogic": [
                "Core Concept A",
                "Triggering Condition B",
                "Business Rule C",
                "Entity State D"
            ]
        }