import unittest
from src.keyword_extraction import extract_keywords

class TestKeywordExtraction(unittest.TestCase):

    def test_extract_keywords(self):
        text = "Python is a high-level programming language. It is easy to learn and widely used."
        keywords = extract_keywords(text, max_ngram_size=2, num_keywords=5)
        self.assertTrue(len(keywords) > 0)
        keyword_texts = [kw[0] for kw in keywords]
        self.assertIn("Python", keyword_texts)

    def test_extract_keywords_empty_text(self):
        text = ""
        keywords = extract_keywords(text, max_ngram_size=2, num_keywords=5)
        self.assertEqual(keywords, [])

if __name__ == '__main__':
    unittest.main()
