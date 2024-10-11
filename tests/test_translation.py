import pytest
from unittest.mock import patch, MagicMock
from src.translation import translate_text, SUPPORTED_LANGUAGES

# Sample data for testing
sample_text = "Hello, how are you?"
src_lang = "english"
tgt_lang = "spanish"

# Mocking the API call for successful translation
def mock_successful_api_call(*args, **kwargs):
    return MagicMock(status_code=200, json=lambda: {"translation_text": "Hola, ¿cómo estás?"})

# Mocking the API call for service unavailable error
def mock_service_unavailable_api_call(*args, **kwargs):
    return MagicMock(status_code=503, json=lambda: {"estimated_time": 5})

# Mocking the API call for model busy error
def mock_model_busy_api_call(*args, **kwargs):
    return MagicMock(status_code=500, text="Model too busy")

# Test successful translation
@patch('requests.post', side_effect=mock_successful_api_call)
def test_translate_text_success(mock_post):
    result = translate_text(sample_text, src_lang, tgt_lang)
    assert result["translation_text"] == "Hola, ¿cómo estás?"
    mock_post.assert_called_once()

# Test unsupported source language
def test_translate_text_unsupported_src_lang():
    with pytest.raises(ValueError, match="Unsupported language code\(s\)"):
        translate_text(sample_text, "unsupported_lang", tgt_lang)

# Test unsupported target language
def test_translate_text_unsupported_tgt_lang():
    with pytest.raises(ValueError, match="Unsupported language code\(s\)"):
        translate_text(sample_text, src_lang, "unsupported_lang")

# Test service unavailable with retries
@patch('requests.post', side_effect=mock_service_unavailable_api_call)
@patch('time.sleep', return_value=None)  # Prevent actual sleeping during tests
def test_translate_text_service_unavailable(mock_sleep, mock_post):
    result = translate_text(sample_text, src_lang, tgt_lang, max_retries=3, retry_delay=1)
    assert result is None
    assert mock_post.call_count == 3  # Should retry 3 times

# Test model busy with retries
@patch('requests.post', side_effect=mock_model_busy_api_call)
@patch('time.sleep', return_value=None)  # Prevent actual sleeping during tests
def test_translate_text_model_busy(mock_sleep, mock_post):
    result = translate_text(sample_text, src_lang, tgt_lang, max_retries=3, retry_delay=1)
    assert result is None
    assert mock_post.call_count == 3  # Should retry 3 times

# Test max retries reached
@patch('requests.post', side_effect=mock_service_unavailable_api_call)
@patch('time.sleep', return_value=None)  # Prevent actual sleeping during tests
def test_translate_text_max_retries_reached(mock_sleep, mock_post):
    result = translate_text(sample_text, src_lang, tgt_lang, max_retries=2, retry_delay=1)
    assert result is None
    assert mock_post.call_count == 2  # Should reach max retries

if __name__ == "__main__":
    pytest.main()
