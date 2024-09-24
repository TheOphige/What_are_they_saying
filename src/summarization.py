# from transformers import pipeline

# def summarize_text(text: str) -> str:
#     """
#     Summarize the input text using a pre-trained transformer model.
#     """
#     summarizer = pipeline('summarization')
#     summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
#     return summary[0]['summary_text']


import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Retrieve API keys from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")


# Initialize the LLM (replace with your preferred model)
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    model_name="mattshumer/reflection-70b:free",
)

# Define the prompt template for summarization
prompt_template = """Summarize the following article content:

{article}

Summary:"""

prompt = PromptTemplate(
    input_variables=["article"],
    template=prompt_template,
)

summarize_chain = prompt | llm | StrOutputParser()

# Function to summarize text or a Wikipedia page
def summarize_input(input_data: str, is_query: bool = True) -> str:
    # Determine whether input_data is a Wikipedia query or text and use the appropriate loader
    if is_query:
        loader = WikipediaLoader(query=input_data)
        # Load the content
        article_content = loader.load()[0].page_content
    else:
        # Load the content
        article_content = input_data
    
    
    # Use the LLM chain to generate a summary
    summary = summarize_chain.invoke(input=article_content) # run(article=article_content)
    
    return summary

# # Example usage
# if __name__ == "__main__":
#     # For Wikipedia query  
#     wikipedia_query = "Artificial Intelligence"
#     summary_from_query = summarize_input(wikipedia_query, is_query=True)
#     print("Summary from Wikipedia:", summary_from_query)

#     # For plain text
#     text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals."
#     summary_from_text = summarize_input(text, is_query=False)
#     print("Summary from Text:", summary_from_text)
