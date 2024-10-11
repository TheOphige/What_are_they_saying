import pytest
from unittest.mock import patch, MagicMock
from src.summarization import summarize_input

# Mock the external dependencies to isolate the function being tested
@pytest.fixture
def mock_wikipedia_loader():
    with patch('summarization.WikipediaLoader') as mock:
        yield mock

@pytest.fixture
def mock_web_loader():
    with patch('summarization.WebBaseLoader') as mock:
        yield mock

@pytest.fixture
def mock_llm_chain():
    with patch('summarization.summarize_chain.invoke') as mock:
        yield mock

def test_summarize_input_wikipedia_query(mock_wikipedia_loader, mock_llm_chain):
    # Setup the mock for WikipediaLoader
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = ["This is the content of the Wikipedia page."]
    mock_wikipedia_loader.return_value = mock_loader_instance
    
    # Setup the mock for the LLM chain
    mock_llm_chain.return_value = "This is the summary."

    # Test the Wikipedia query input
    summary = summarize_input("Python (programming language)", "Wikipedia Query")
    assert summary == "This is the summary."
    mock_wikipedia_loader.assert_called_once_with(query="Python (programming language)", load_max_docs=6)
    mock_llm_chain.assert_called_once()

def test_summarize_input_text(mock_llm_chain):
    # Setup the mock for the LLM chain
    mock_llm_chain.return_value = "This is the summary."

    # Test the text input
    summary = summarize_input("This is a test text.", "Text")
    assert summary == "This is the summary."
    mock_llm_chain.assert_called_once()

def test_summarize_input_url(mock_web_loader, mock_llm_chain):
    # Setup the mock for WebBaseLoader
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = ["This is the content from the URL."]
    mock_web_loader.return_value = mock_loader_instance
    
    # Setup the mock for the LLM chain
    mock_llm_chain.return_value = "This is the summary from the URL."

    # Test the URL input
    summary = summarize_input("http://example.com", "URL")
    assert summary == "This is the summary from the URL."
    mock_web_loader.assert_called_once_with("http://example.com")
    mock_llm_chain.assert_called_once()

def test_summarize_input_invalid_type(mock_llm_chain):
    # Test with an invalid input type
    with pytest.raises(ValueError, match="Invalid input type provided."):
        summarize_input("This should fail.", "Invalid Type")

if __name__ == "__main__":
    pytest.main()
