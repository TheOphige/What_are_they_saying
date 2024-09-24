import streamlit as st
from src.data_processing import clean_text
from src.translation import translate_text
from src.summarization import summarize_input
from src.keyword_extraction import extract_keywords
from src.visualization import generate_word_cloud


# Sidebar for user input
st.sidebar.header("What are they saying?")
input_type = st.sidebar.selectbox("Choose input type", ("Text", "Wikipedia Query"))
input_text = st.sidebar.text_area("Enter text or Wikipedia Query here")
# language
st.sidebar.title("Select Language")
source_language = st.sidebar.selectbox("Source Language", ["en", "fr", "de", "es"])
target_language = st.sidebar.selectbox("Target Language", ["en", "fr", "de", "es"])
# to_do
st.sidebar.title("Select Options")
summarize = st.sidebar.checkbox("Summarize")
keywords = st.sidebar.checkbox("Keywords")
chat = st.sidebar.checkbox("Chat") 

if st.sidebar.button("Analyze"):
    if not input_text:
        st.error("Please enter text or provide a URL.")
    else:
        try:
            input_text = clean_text(input_text)

            if summarize:
                # Summarization
                with st.spinner("Summarizing the long talks..."):
                    if input_type == "Wikipedia Query":
                        summary = summarize_input(input_text, is_query=True)
                    if input_type == "Text":
                        summary = summarize_input(input_text, is_query=False)
                    if not summary:
                        st.error("Summarization failed. Please try again later.")
                        st.stop()


                # Translation
                if source_language == target_language:
                    translated_summary = summary
                else:
                    with st.spinner("Translating to your language..."):
                        translated_summary = translate_text(summary, source_language, target_language)
                        translated_summary = translated_summary[0]["translation_text"]
                        if not translated_summary:
                            st.error("Translation failed. Please try a different text or check your input.")
                            st.stop()
                
                # Display translated Summary
                with st.expander("summary"):
                    st.write("#### Summary in your language")
                    st.write(translated_summary)
            
            if keywords:
                # Keyword Extraction
                with st.spinner("Extracting keywords..."):
                    keywords = extract_keywords(translated_summary, max_ngram_size=3, num_keywords=5, language=target_language)
                    if not keywords:
                        st.warning("No keywords were extracted from the summary.")
                
                # keywords
                with st.expander("keywords"):
                    # Display Keywords
                    st.write("### Keywords")
                    for kw, score in keywords:
                        st.write(f"{kw}: {score}")
                    
                    # Display Word Cloud
                    st.write("### Word Cloud")
                    generate_word_cloud(keywords)

            if chat:
                # Chat with text / Ask questions
                st.write("chatting..")
            
            if not any([chat, keywords, summarize]):
                st.info("Select at least one option: \"Summarize or Keywords or Chat")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")