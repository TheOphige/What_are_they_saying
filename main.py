import streamlit as st
from src.data_processing import clean_text
from src.translation import translate_text
from src.summarization import summarize_input
from src.keyword_extraction import extract_keywords, convert_keywords
from src.visualization import generate_word_cloud


# Sidebar for user input
st.sidebar.header("What are they saying?")
input_type = st.sidebar.selectbox("Choose input type", ("Wikipedia Query", "Text"))
input_text = st.sidebar.text_area("Enter text or Search Query here")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", options=["Summarize in language", "Chat with doc"], index=0)

if mode == "Summarize in language":
    # language
    source_language = st.sidebar.selectbox("Source Language", ["en", "fr", "de", "es"])
    target_language = st.sidebar.selectbox("Target Language", ["en", "fr", "de", "es"])


    if st.sidebar.button("Analyze"):
        if not input_text:
            st.error("Please enter text or provide a URL.")
        else:
            try:
                input_text = clean_text(input_text)

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
                
                # Keyword Extraction
                with st.spinner("Extracting keywords..."):
                    keywords = extract_keywords(translated_summary, max_ngram_size=3, num_keywords=15, language=target_language)
                    if not keywords:
                        st.warning("No keywords were extracted from the summary.")
                
                # Display Keywords
                with st.expander("keywords"):
                    st.write("### Keywords")
                    keywords_string = ', '.join([keyword for keyword, score in keywords])
                    st.write(keywords_string)
                        
                    # Display Word Cloud
                    st.write("### Word Cloud")
                    converted_keywords = convert_keywords(keywords)
                    print(converted_keywords)
                    generate_word_cloud(converted_keywords)


            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")



elif mode == "Chat with doc":
    # Chat with text / Ask questions
    if st.sidebar.button("Analyze"):
        if not input_text:
            st.error("Please enter text or provide a URL.")
        else:
            try:
                st.write("chatting..")


            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")