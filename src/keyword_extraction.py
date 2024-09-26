import yake

def extract_keywords(text: str, max_ngram_size: int = 3, num_keywords: int = 20, language: str = 'en') -> list:
    """
    Extract keywords from the text using YAKE (Yet Another Keyword Extractor).
    
    Args:
        text (str): The input text to extract keywords from.
        max_ngram_size (int): The maximum size of the n-grams (default is 3).
        num_keywords (int): The number of top keywords to extract (default is 10).
        language (str): The language of the text (default is 'en' for English).
    
    Returns:
        list: A list of extracted keywords with their relevance scores.
    """
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return keywords

def convert_keywords(keywords):
    converted_keywords = []
    for word, freq in keywords:
        # Replace spaces with underscores and multiply frequencies by 100
        new_word = word.replace(' ', '_')
        new_freq = round(freq * 1000, 2)  # Multiply by 100 and round to 2 decimal places
        converted_keywords.append((new_word, new_freq))
    return converted_keywords

# # Example usage:
# text = "This is an example text for keyword extraction."
# language = "en"  # Change to any language code, e.g., 'fr', 'es', 'de', etc.
# keywords = extract_keywords(text, max_ngram_size=3, num_keywords=5, language=language)
# print(keywords)
