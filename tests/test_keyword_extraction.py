import pytest
from src.keyword_extraction import extract_keywords, convert_keywords

def test_extract_keywords_valid_input():
    text = "Natural language processing and keyword extraction are important tasks in AI."
    keywords = extract_keywords(text, max_ngram_size=2, num_keywords=5)
    assert isinstance(keywords, list)
    assert len(keywords) <= 5
    assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)

def test_extract_keywords_empty_input():
    keywords = extract_keywords("")
    assert keywords == []

def test_extract_keywords_long_input():
    text = " ".join(["word"] * 1000)  # Repeated word to test handling of long input
    keywords = extract_keywords(text, num_keywords=5)
    assert len(keywords) == 1
    assert keywords[0][0] == "word"

def test_convert_keywords_valid_input():
    keywords = [("natural language", 0.5), ("keyword extraction", 0.3)]
    converted = convert_keywords(keywords)
    assert isinstance(converted, list)
    assert len(converted) == 2
    assert converted[0] == ("natural_language", 50.0)
    assert converted[1] == ("keyword_extraction", 30.0)

def test_convert_keywords_empty_input():
    converted = convert_keywords([])
    assert converted == []

def test_convert_keywords_single_keyword():
    keywords = [("hello world", 0.123)]
    converted = convert_keywords(keywords)
    assert converted[0] == ("hello_world", 12.3)

def test_extract_keywords_invalid_language():
    text = "Testing invalid language handling."
    with pytest.raises(ValueError):
        extract_keywords(text, language="invalid_language")

if __name__ == "__main__":
    pytest.main()
