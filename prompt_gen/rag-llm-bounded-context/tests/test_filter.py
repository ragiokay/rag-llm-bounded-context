import pytest
from unittest.mock import MagicMock
from filters.DomainEventsFilter import DomainEventsFilter

def test_domain_events_filter_cleans_data():
    """
    Test whether the filter can correctly filter out noise and extract only the 'output' field.
    """
    # 1. Create a "fake" Retriever (Mock Object)
    mock_retriever = MagicMock()
    
    # 2. The simulated database returned "raw dirty data".
    mock_retriever.search.return_value = [
        {"distance": 0.99, "input": "text1", "output": {"DomainEvents": ["A"]}},
        {"distance": 0.85, "input": "text2", "output": {"DomainEvents": ["B"]}},
        {"distance": 0.11, "input": "text3"}  # Intentionally insert a defective product that produces no output.
    ]
    
    # 3. Feed the fake Retriever to the Filter
    filter_obj = DomainEventsFilter(mock_retriever)
    clean_results = filter_obj.get_clean_examples("dummy query", top_k=3)
    
    # 4. verify
    assert len(clean_results) == 2  # The third defective product should be automatically filtered out.
    assert clean_results[0] == {"DomainEvents": ["A"]}
    assert clean_results[1] == {"DomainEvents": ["B"]}
    
    # verify Filter call the correct data table (maven_ere_causal)
    mock_retriever.search.assert_called_with("dummy query", "maven_ere_causal", 3)