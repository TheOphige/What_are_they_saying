import pytest
from src.data_processing import clean_text, scrape_wikipedia_page

def test_removes_extra_spaces():
    input_text = "This    is  a    test."
    expected_output = "This is a test."
    assert clean_text(input_text) == expected_output

def test_removes_leading_and_trailing_spaces():
    input_text = "   Hello World!   "
    expected_output = "Hello World!"
    assert clean_text(input_text) == expected_output

def test_handles_empty_string():
    input_text = ""
    expected_output = ""
    assert clean_text(input_text) == expected_output

def test_handles_only_spaces():
    input_text = "       "
    expected_output = ""
    assert clean_text(input_text) == expected_output

def test_preserves_single_spaces_between_words():
    input_text = "  Clean   this   text.  "
    expected_output = "Clean this text."
    assert clean_text(input_text) == expected_output

def test_removes_newlines():
    input_text = "Line 1.\nLine 2."
    expected_output = "Line 1. Line 2."
    assert clean_text(input_text) == expected_output

def test_removes_tabs():
    input_text = "This\tis\ta\ttest."
    expected_output = "This is a test."
    assert clean_text(input_text) == expected_output

# You can add tests for scrape_wikipedia_page here as needed.

if __name__ == "__main__":
    pytest.main()
