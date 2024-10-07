__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import chromadb

# # Load environment variables from .env
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = st.secrets["OPENROUTER_BASE_URL"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = ChatOpenAI(
                openai_api_key=OPENROUTER_API_KEY,  # Ensure the key is passed correctly
                openai_api_base=OPENROUTER_BASE_URL,  # Ensure the base URL is correct
                model_name="mistralai/pixtral-12b:free"  # Adjust based on model availability
            )


# retrieve artcle
def setup_retriever(input_type: str, input_text: None):
    if input_type == "Wikipedia Query":
        retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

    elif input_type == "Text":
        data = input_text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_text(data)

        embeddings = HuggingFaceInferenceAPIEmbeddings(
                        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
                    )

        vectorstore = Chroma.from_texts(texts=all_splits, embedding=embeddings)

        retriever = vectorstore.as_retriever(k=4)

    elif input_type == "URL":
        loader = WebBaseLoader(input_text)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        embeddings = HuggingFaceInferenceAPIEmbeddings(
                        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
                    )
        
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

        retriever = vectorstore.as_retriever(k=4)

    return retriever



# prompts for llm
def setup_prompt(input_type: str):
    if input_type == "Wikipedia Query":
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
            "Remember if the latest user question does not reference context,"
            "return as is."
        )

        # Create a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Answer question prompt
        # This system prompt helps the AI understand that it should provide concise answers
        # based on the retrieved context and indicates what to do if the answer is unknown
        qa_system_prompt = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, 
        answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
        You don't need to return user question.

        Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
        justifies the answer and the LINK of article. Return a citation for every quote across all articles 
        that justify the answer. 


        Here are the Wikipedia articles:{context}

        Answer:
        Citations:
        """

        # Create a prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    elif input_type == "Text":
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
            "Remember if the latest user question does not reference context,"
            "return as is."
        )

        # Create a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Answer question prompt
        # This system prompt helps the AI understand that it should provide concise answers
        # based on the retrieved context and indicates what to do if the answer is unknown
        qa_system_prompt = """You're a helpful AI assistant. Given a user question and some article snippets, 
        answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
        You don't need to return user question.

        Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
        justifies the answer and the LINK of article. Return a citation for every quote across all articles 
        that justify the answer. 


        Here are the articles:{context}

        Answer:
        Citations:
        """

        # Create a prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    elif input_type == "URL":
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
            "Remember if the latest user question does not reference context,"
            "return as is."
        )

        # Create a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Answer question prompt
        # This system prompt helps the AI understand that it should provide concise answers
        # based on the retrieved context and indicates what to do if the answer is unknown
        qa_system_prompt = """You're a helpful AI assistant. Given a user question and some article snippets, 
        answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
        You don't need to return user question.

        Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
        justifies the answer and the LINK of article. Return a citation for every quote across all articles 
        that justify the answer. 


        Here are the articles:{context}

        Answer:
        Citations:
        """

        # Create a prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    return contextualize_q_prompt, qa_prompt



def on_chat_submit(input_type, input_text, chat_input):
    # clear chromadb cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Display raw chat input for user
    st.session_state.history.append({"role": "user", "content": chat_input})

    # retrieve chat history
    chat_history = st.session_state.conversation_history

    chat_input = chat_input.strip().lower()
    try:
        retriever = setup_retriever(input_type, input_text)
        contextualize_q_prompt, qa_prompt = setup_prompt(input_type)

        # Create a history-aware retriever
        # This uses the LLM to help reformulate the question based on chat history
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Create a chain to combine documents for question answering
        # `create_stuff_documents_chain` feeds all retrieved context into the LLM
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create a retrieval chain that combines the history-aware retriever and the question answering chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": chat_input, "chat_history": chat_history})

        # Append assistant's response to the conversation history
        st.session_state.conversation_history.append(HumanMessage(content=chat_input))
        st.session_state.conversation_history.append(AIMessage(content=result["answer"]))
        st.session_state.history.append({"role": "assistant", "content": result["answer"]})
    
    except Exception as e:
        st.error(f"Error occurred: {e}")



def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    else:
        st.session_state.history = []

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    else:
        st.session_state.conversation_history = []


def initialize_message(input_type, input_text):
    if input_type == "Wikipedia Query":
        welcome_text = f"""Hello! I am WATS. How can I assist you today?
        \nI noticed you've put in a Wikipedia Query.
        What would you like to know about {input_text} or any other thing from wikipedia?
        """
        st.session_state.history.append({"role": "assistant", "content": welcome_text})
    elif input_type == "Text":
        welcome_text = """Hello! I am WATS. How can I assist you today?
        \nI noticed you've put in a Text.
        What would you like to know about the Text (i.e article content you've pasted)?
        """
        st.session_state.history.append({"role": "assistant", "content": welcome_text})
    elif input_type == "URL":
        welcome_text = f"""Hello! I am WATS. How can I assist you today?
        \nI noticed you've put in the link to an article.
        What would you like to know about the article from {input_text} ?
        """
        st.session_state.history.append({"role": "assistant", "content": welcome_text})


# # Streamlit Interface
# st.title("WATS: Question Answering Agent")

# # Sidebar for user input
# st.sidebar.header("What are they saying?")
# input_type = st.sidebar.selectbox("Choose input type:", ("Wikipedia Query", "Text", "URL"))
# input_text = st.sidebar.text_area("Enter text, search query, or URL:")

# # Sidebar for Mode Selection
# mode = st.sidebar.radio("Select Mode:", options=["Summarize in language", "Chat with doc"], index=0)

# if mode == "Summarize in language":
#     pass

# elif mode == "Chat with doc":
#     try:
#         if st.sidebar.button("Analyze"):
#             if not input_text:
#                 st.error("Please provide valid input.")
#             else:
#                 initialize_session_state()
#                 if not st.session_state.history:
#                     st.session_state.history.append({"role": "assistant", "content": "Hello! I am WATS. How can I assist you today?"})
#                     st.session_state.conversation_history = []

#         # Handle chat input
#         chat_input = st.chat_input("Ask me a question about the article:")
#         if chat_input:
#             on_chat_submit(input_type, input_text, chat_input)

#         # Display chat history
#         for message in st.session_state.history:
#             role = message["role"]
#             avatar_image = "imgs/avatar_streamly.png" if role == "assistant" else "imgs/stuser.png"
#             with st.chat_message(role, avatar=avatar_image):
#                 st.write(message["content"])
#     except Exception as e:
#         # st.error(f"Error occurred: {e}")
#         st.info("Please enter text, search query, or URL. Then click ANALYZE.")