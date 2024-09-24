import yake

def extract_keywords(text: str, max_ngram_size: int = 3, num_keywords: int = 10, language: str = 'en') -> str:
    """
    Extract keywords from the text using YAKE (Yet Another Keyword Extractor).
    
    Args:
        text (str): The input text to extract keywords from.
        max_ngram_size (int): The maximum size of the n-grams (default is 3).
        num_keywords (int): The number of top keywords to extract (default is 10).
        language (str): The language of the text (default is 'en' for English).
    
    Returns:
        str: A string of extracted keywords separated by commas.
    """
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    
    # Extract the keyword part only and join with commas
    keyword_list = [keyword for keyword, score in keywords]
    return ', '.join(keyword_list)

# Example usage
# text = "Bezos fundó Amazon y es considerado el hombre más rico del mundo, con un valor neto de millardos de dólares."
# keywords = extract_keywords(text, language='es')
# print(keywords)
