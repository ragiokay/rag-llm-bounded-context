from prompt_generator.BasePromptGenerator import BasePromptGenerator

class DomainEventsPromptGenerator(BasePromptGenerator):
    def get_task(self) -> str:
        return "Analyze the business logic, identify all state changes (including initial triggers), and CONVERT them into Domain Events."

    def get_rules(self) -> str:
        return (
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY a valid JSON object matching the exact format of the schema. No markdown, no tags, no explanations.\n"
            "2. Convert every action or trigger into a PAST TENSE verb in PascalCase.\n"
            "3. Examples of valid events: 'CartSubmitted', 'DiscountCalculated', 'InvoiceGenerated'."
        )

    def get_output_schema(self) -> dict:
        return {
            "DomainEvents": [
                "ConceptOneCompleted",
                "StateTwoChanged",
                "ActionThreeExecuted"
            ]
        }