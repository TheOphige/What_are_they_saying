from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(src_lang: str, tgt_lang: str):
    """
    Load the MarianMT translation model for a specific language pair.
    """
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text: str, model, tokenizer) -> str:
    """
    Translate text from the source language to the target language.
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text
