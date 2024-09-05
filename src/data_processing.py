import re
import requests
from bs4 import BeautifulSoup
import wikipediaapi

def scrape_wikipedia_page(url: str) -> str:
    """
    Scrape the content from a Wikipedia page given its URL.
    """
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_name = url.split('/')[-1]
    page = wiki_wiki.page(page_name)
    return page.text

def clean_text(text: str) -> str:
    """
    Clean the input text by removing unwanted characters, extra spaces, and formatting issues.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def get_text_from_url(url: str) -> str:
    """
    Get text content from a given URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return clean_text(text)
    except Exception as e:
        print(f"Error scraping URL: {e}")
        return ""
