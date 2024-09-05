import streamlit as st
from src.data_processing import scrape_wikipedia_page, clean_text
from src.translation import load_translation_model, translate_text
from src.summarization import summarize_text
from src.keyword_extraction import extract_keywords
from src.visualization import generate_word_cloud

# Sidebar for user input
st.sidebar.header("What are they saying?")
input_type = st.sidebar.selectbox("Choose input type", ("Text", "Wikipedia URL"))
input_text = st.sidebar.text_area("Enter text or URL here")
source_language = st.sidebar.selectbox("Source Language", ["en", "fr", "de", "es"])
target_language = st.sidebar.selectbox("Target Language", ["en", "fr", "de", "es"])

if st.sidebar.button("Analyze"):
    if not input_text:
        st.error("Please enter text or provide a URL.")
    else:
        try:
            if input_type == "Wikipedia URL":
                input_text = scrape_wikipedia_page(input_text)
                if not input_text:
                    st.error("Failed to retrieve content from the provided URL. Please check the URL and try again.")
                    st.stop()
            else:
                input_text = clean_text(input_text)

            model, tokenizer = load_translation_model(source_language, target_language)
            if model is None or tokenizer is None:
                st.error("Failed to load translation model. Please check the model configurations.")
                st.stop()
                
            translated_text = translate_text(input_text, model, tokenizer)
            if not translated_text:
                st.error("Translation failed. Please try a different text or check your input.")
                st.stop()
            
            summary = summarize_text(translated_text)
            if not summary:
                st.error("Summarization failed. Please try again later.")
                st.stop()
            
            keywords = extract_keywords(summary)
            if not keywords:
                st.warning("No keywords were extracted from the summary.")
            
            st.write("### Summary")
            st.write(summary)
            
            st.write("### Keywords")
            for kw, score in keywords:
                st.write(f"{kw}: {score}")
            
            st.write("### Word Cloud")
            generate_word_cloud(keywords)
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
