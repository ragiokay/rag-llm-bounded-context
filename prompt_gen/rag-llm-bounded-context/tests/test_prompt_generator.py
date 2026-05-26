import pytest
from prompt_generator.PromptBuilder import PromptBuilder

def test_prompt_builder_under_limit():
    """
    Test Plan TC 2.1.1: Within the token restrictions, the system was able to correctly receive the target elements and business text during testing.
    """
    builder = PromptBuilder()
    examples = [{"output": {"DomainEvents": ["OrderPlaced"]}}]
    
    builder.add_rag_examples(examples)
    
    assert "[RAG EXAMPLES" in builder.prompt
    assert "OrderPlaced" in builder.prompt

def test_prompt_builder_exceeds_limit():
    """
    Test Plan TC 2.1.2: The test showed that the system correctly truncates the Context when its length is intentionally exceeded by the Token limit.
    """
    builder = PromptBuilder()
    
    # Create two cases: the first is normal, the second is extra huge.
    normal_example = [{"output": {"DomainEvents": ["NormalEvent"]}}]
    huge_example = [{"output": {"DomainEvents": ["A" * (builder.MAX_RAG_CHARS + 100)]}}]
    
    # Build both cases in together
    builder.add_rag_examples(normal_example + huge_example)
    
    # Verify: The first normal case has been added.
    assert "NormalEvent" in builder.prompt
    # Verify: The second major case was blocked by the system, successfully protecting the LLM.
    assert "A" * 100 not in builder.prompt