import pytest
from unittest.mock import patch, MagicMock
from src.visualization import upload_to_imgur, generate_word_cloud

# Sample data for testing
sample_keywords = [("word1", 1), ("word2", 2), ("word3", 3)]

# Mocking the Imgur API response for successful upload
def mock_successful_imgur_upload(*args, **kwargs):
    return MagicMock(status_code=200, json=lambda: {"data": {"link": "http://example.com/image.png"}})

# Mocking the Imgur API response for failed upload
def mock_failed_imgur_upload(*args, **kwargs):
    return MagicMock(status_code=400, json=lambda: {"data": {"error": "Bad Request"}})

# Test successful upload to Imgur
@patch('requests.post', side_effect=mock_successful_imgur_upload)
@patch('streamlit.secrets', new_callable=MagicMock)
def test_upload_to_imgur_success(mock_secrets, mock_post):
    mock_secrets.return_value = {"IMGUR_CLIENT_ID": "test_client_id"}
    
    with open("test_image.png", "wb") as img_file:
        img_file.write(b"fake_image_data")  # Create a fake image file
    
    with open("test_image.png", "rb") as img_file:
        result = upload_to_imgur(img_file)
    
    assert result == "http://example.com/image.png"
    mock_post.assert_called_once()

# Test failed upload to Imgur
@patch('requests.post', side_effect=mock_failed_imgur_upload)
@patch('streamlit.secrets', new_callable=MagicMock)
def test_upload_to_imgur_failure(mock_secrets, mock_post):
    mock_secrets.return_value = {"IMGUR_CLIENT_ID": "test_client_id"}
    
    with open("test_image.png", "wb") as img_file:
        img_file.write(b"fake_image_data")  # Create a fake image file
    
    with open("test_image.png", "rb") as img_file:
        result = upload_to_imgur(img_file)
    
    assert result is None
    mock_post.assert_called_once()

# Test generating the word cloud
@patch('visualization.upload_to_imgur', return_value="http://example.com/image.png")
def test_generate_word_cloud(mock_upload):
    result = generate_word_cloud(sample_keywords)
    assert result is None  # The function does not return anything, we check for side effects
    mock_upload.assert_called_once()

if __name__ == "__main__":
    pytest.main()
