import unittest
from src.data_processing import clean_text, scrape_wikipedia_page

class TestDataProcessing(unittest.TestCase):

    def test_clean_text(self):
        text = "  This is a   test string.  "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "This is a test string.")

    def test_scrape_wikipedia_page(self):
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        content = scrape_wikipedia_page(url)
        self.assertIn("Python", content)

    def test_scrape_invalid_url(self):
        url = "https://en.wikipedia.org/wiki/Nonexistent_Page"
        content = scrape_wikipedia_page(url)
        self.assertEqual(content, "")

if __name__ == '__main__':
    unittest.main()
