import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
# from dotenv import find_dotenv, load_dotenv

# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = st.secrets["OPENROUTER_BASE_URL"]

# Initialize the LLM (replace with your preferred model)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="microsoft/mai-ds-r1:free",
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
def summarize_input(input_text, input_type) -> str:
    # Determine whether input_data is a Wikipedia query or text and use the appropriate loader
    if input_type == "Wikipedia Query":
        loader = WikipediaLoader(query=input_text, load_max_docs= 6) #, load_all_available_meta= True, doc_content_chars_max=2000)
        # Load the content
        article_content = loader.load() #[0].page_content
    elif input_type == "Text":
        article_content = input_text
    elif input_type == "URL":
        loader = WebBaseLoader(input_text)
        # Load the content
        article_content = loader.load()
    
    
    # Use the LLM chain to generate a summary
    summary = summarize_chain.invoke(input=article_content) # run(article=article_content)
    
    return summary