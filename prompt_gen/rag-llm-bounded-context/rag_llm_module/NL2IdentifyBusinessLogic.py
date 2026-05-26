from rag_llm_module.BaseNL2Service import BaseNL2Service
from prompt_generator.BasePromptGenerator import BasePromptGenerator
from prompt_generator.BusinessLogicPromptGenerator import BusinessLogicPromptGenerator

class NL2IdentifyBusinessLogic(BaseNL2Service):
    def get_generator(self) -> BasePromptGenerator:
        return BusinessLogicPromptGenerator()