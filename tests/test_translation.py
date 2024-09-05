import unittest
from src.translation import load_translation_model, translate_text

class TestTranslation(unittest.TestCase):

    def test_load_translation_model(self):
        model, tokenizer = load_translation_model('en', 'fr')
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

    def test_translate_text(self):
        model, tokenizer = load_translation_model('en', 'fr')
        text = "Hello, how are you?"
        translated = translate_text(text, model, tokenizer)
        self.assertEqual(translated.lower(), "bonjour, comment ça va ?")

    def test_translate_invalid_text(self):
        model, tokenizer = load_translation_model('en', 'fr')
        text = ""
        translated = translate_text(text, model, tokenizer)
        self.assertEqual(translated, "")

if __name__ == '__main__':
    unittest.main()
