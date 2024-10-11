import streamlit as st
from src.data_processing import clean_text
from src.translation import translate_text, SUPPORTED_LANGUAGES
from src.summarization import summarize_input
from src.keyword_extraction import extract_keywords, convert_keywords
from src.visualization import generate_word_cloud
from src.wats_chat import initialize_session_state, on_chat_submit, initialize_message

# Streamlit Page Configuration
st.set_page_config(
    page_title="WATS - An Intelligent Question Answering Agent",
    page_icon="imgs/avatar_wats.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/TheOphige/What_are_they_saying",
        "Report a bug": "https://github.com/TheOphige/What_are_they_saying",
        "About": """
            ## WATS: Question Answering Agent
            ### Powered using Mistral-

            **GitHub**: https://github.com/TheOphige/What_are_they_saying

            The AI Assistant named, WATS, aims to help you understand any article,
            it summarizes the article and enable you to chat with the article,
            it answers any questions you might have about the article.
        """
    }
)

# Streamlit Title
st.title("WATS: Question Answering Agent")

# Sidebar for user input
st.sidebar.header("What Are They Saying?")
input_type = st.sidebar.selectbox("Choose input type:", ("Wikipedia Query", "Text", "URL"))
input_text = st.sidebar.text_area("Enter text, search query, or URL:")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", options=["Summarize in language", "Chat with article"], index=0)

# Summarize article in user language
if mode == "Summarize in language":
    # language
    source_language = st.sidebar.selectbox("Source Language:", list(SUPPORTED_LANGUAGES.keys()), index=3)
    target_language = st.sidebar.selectbox("Target Language:", list(SUPPORTED_LANGUAGES.keys()), index=3)


    if st.sidebar.button("Summarize"):
        if not input_text:
            st.error("Please enter text, search query, or URL.")
        else:
            try:
                input_text = clean_text(input_text)

                # Summarization
                with st.spinner("Summarizing the long talks..."):
                    summary = summarize_input(input_text, input_type)

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
                    # print(converted_keywords)
                    generate_word_cloud(converted_keywords)


            except Exception as e:
                st.info("Please enter text, search query, or URL. Then click SUMMARIZE.")
                st.error(f"An unexpected error occurred: {e}")


# chat with article
elif mode == "Chat with article":
    try:
        if st.sidebar.button("Chat"):
            if not input_text:
                st.error("Please provide valid input.")
            else:
                initialize_session_state()

                initialize_message(input_type, input_text)
            
        # Handle chat input
        chat_input = st.chat_input("Ask me a question about the article:")

        

        if chat_input:
            with st.spinner("🛠 Working..."):
                on_chat_submit(input_type, input_text, chat_input)

        # Display chat history
        for message in st.session_state.history:
            role = message["role"]
            avatar_image = "imgs/avatar_wats.png" if role == "assistant" else "imgs/wats_user.png"
            with st.chat_message(role, avatar=avatar_image):
                st.write(message["content"])
    except Exception as e:
        # st.error(f"Error occurred: {e}")
        st.info("Please enter text, search query, or URL. Then click CHAT.")