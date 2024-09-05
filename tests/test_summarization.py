import unittest
from src.summarization import summarize_text

class TestSummarization(unittest.TestCase):

    def test_summarize_text(self):
        text = ("Python is an interpreted, high-level, general-purpose programming language. "
                "Python's design philosophy emphasizes code readability with its notable use of significant indentation.")
        summary = summarize_text(text, max_length=50, min_length=10)
        self.assertTrue(len(summary) <= 50)
        self.assertTrue("Python" in summary)

    def test_summarize_empty_text(self):
        text = ""
        summary = summarize_text(text, max_length=50, min_length=10)
        self.assertEqual(summary, "")

if __name__ == '__main__':
    unittest.main()
