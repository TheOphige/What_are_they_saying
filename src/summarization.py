from transformers import pipeline

def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """
    Summarize the input text using a pre-trained transformer model.
    """
    summarizer = pipeline('summarization')
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']
