import yake

def extract_keywords(text: str, max_ngram_size: int = 3, num_keywords: int = 10) -> list:
    """
    Extract keywords from the text using YAKE (Yet Another Keyword Extractor).
    """
    kw_extractor = yake.KeywordExtractor(n=max_ngram_size, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return keywords
