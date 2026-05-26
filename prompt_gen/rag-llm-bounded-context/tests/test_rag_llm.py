import pytest
from unittest.mock import patch, MagicMock
from rag_llm_module.ResponseParser import ResponseParser
from rag_llm_module.NL2IdentifyDomainEvents import NL2IdentifyDomainEvents

# ==========================================
# Extreme boundary testing of parser (Edge Cases)
# ==========================================

def test_parser_valid_json_with_markdown():
    """
    testing LLaMA 3.2 1B: strings enclosed in Markdown ```json ```.
    """
    parser = ResponseParser()
    markdown_output = '''
    Here are your domain events:
    ```json
    {
      "DomainEvents": ["OrderPlaced", "PaymentProcessed"]
    }
    ```
    Hope this helps!
    '''
    
    result = parser.sanitize_json(markdown_output)
    
    # verify
    assert "DomainEvents" in result
    assert result["DomainEvents"] == ["OrderPlaced", "PaymentProcessed"]

def test_parser_smart_scanner_with_noise():
    """
    Test Plan TC 2.2.1 : Testing whether the JSONDecoder scanner can ignore nonsense.
    """
    parser = ResponseParser()
    noisy_output = """
    Sure, here is your logic!
    {"DomainEvents": ["PaymentProcessed"]}
    And here is some random stuff I invented:
    {"INPUT": "Fake Data", "OrderId": 123}
    """
    
    result = parser.sanitize_json(noisy_output)
    
    # Verification: DomainEvents were successfully captured, and fake data INPUT was perfectly discarded.
    assert "DomainEvents" in result
    assert result["DomainEvents"][0] == "PaymentProcessed"
    assert "INPUT" not in result

def test_parser_valid_json_but_wrong_schema():
    """
    Test: The model outputs perfect JSON, but it's not the structure we need at all (no BusinessLogic or DomainEvents).
    """
    parser = ResponseParser()
    wrong_schema_output = '{"Hello": "World", "Status": "Success"}'
    
    result = parser.sanitize_json(wrong_schema_output)
    
    # Because the correct key could not be found, an Invalid error was returned.
    assert "error" in result
    assert result["error"] == "Invalid JSON format"

def test_parser_completely_hallucinated_output():
    """
    Test Plan TC 2.2.2: A string that is not JSON at all (Hallucination)
    """
    parser = ResponseParser()
    hallucinated_output = "I am an AI. I don't want to output JSON today. Here is a poem instead..."
    
    result = parser.sanitize_json(hallucinated_output)
    
    assert "error" in result
    assert result["error"] == "Invalid JSON format"
