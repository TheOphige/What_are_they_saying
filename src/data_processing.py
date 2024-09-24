import re
import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_page(url: str) -> str:
    headers = {
        "User-Agent": "WhatAreTheySaying/1.0 (youremail@example.com)"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.text

def clean_text(text: str) -> str:
    """
    Clean the input text by removing unwanted characters, extra spaces, and formatting issues.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def get_text_from_url(url: str) -> str:
    """
    Fetches and cleans the main text content from a given URL.
    Args:
        url (str): The URL of the page.
    Returns:
        str: Cleaned text content from the page.
    """
    try:
        headers = {
            "User-Agent": "WhatAreTheySaying/1.0 (youremail@example.com)"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')

        # Extract and clean text from paragraphs
        text = ' '.join([para.get_text() for para in paragraphs])

        # Check if any content was extracted
        if not text.strip():
            raise ValueError("No content extracted from the URL")

        return clean_text(text)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching URL content: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error occurred while processing the URL: {e}")
