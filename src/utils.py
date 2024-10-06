# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
# from langchain.docstore.document import Document
# from langchain_community.vectorstores import FAISS
# import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from dotenv import find_dotenv, load_dotenv
# import os
# import logging


# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Load environment variables
# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Initialize session state
# def initialize_session_state():
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     if 'conversation_history' not in st.session_state:
#         st.session_state.conversation_history = []
#     if 'retriever' not in st.session_state:
#         st.session_state.retriever = None

# # Initialize the conversation with a greeting
# def initialize_conversation():
#     conversation_history = [
#         {"role": "system", "content": "You are WATS, a specialized AI assistant."},
#         {"role": "system", "content": "Answer questions based on the given article, Wikipedia query, or URL. Reference the relevant text."},
#         {"role": "assistant", "content": "Hello! How can I assist you today?"}
#     ]
#     return conversation_history


# def setup_retriever(article_content):
#     embeddings = HuggingFaceInferenceAPIEmbeddings(
#                     api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
#                 )

    
#     # Create a FAISS index using both the text embeddings and the embeddings instance
#     index = faiss.IndexFlatL2(len(embeddings.embed_query(article_content)))

#     vector_store = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={},
#     )
    
#     retriever = vector_store.as_retriever()
#     # Return the retriever
#     return retriever




# # Handle user queries
# def on_chat_submit(chat_input):
#     if 'conversation_history' not in st.session_state or st.session_state.retriever is None:
#         st.error("No conversation initialized. Please provide article content first.")
#         return

#     user_input = chat_input.strip().lower()
#     st.session_state.conversation_history.append({"role": "user", "content": user_input})

#     try:
#         # Use the retriever to extract relevant parts of the document
#         relevant_docs = st.session_state.retriever.get_relevant_documents(user_input)

#         if relevant_docs:
#             llm = ChatOpenAI(
#                 openai_api_key=OPENROUTER_API_KEY,  # Ensure the key is passed correctly
#                 openai_api_base=OPENROUTER_BASE_URL,  # Ensure the base URL is correct
#                 model_name="mistralai/pixtral-12b:free"  # Adjust based on model availability
#             )
#             qa_chain = load_qa_chain(llm, chain_type="stuff")
#             answer = qa_chain.run(relevant_docs, question=user_input)

#             # Append assistant's response to the conversation history
#             st.session_state.conversation_history.append({"role": "assistant", "content": answer})
#             st.session_state.history.append({"role": "user", "content": user_input})
#             st.session_state.history.append({"role": "assistant", "content": answer})
#         else:
#             no_info_msg = "The information was not provided in the text."
#             st.session_state.conversation_history.append({"role": "assistant", "content": no_info_msg})
#             st.session_state.history.append({"role": "assistant", "content": no_info_msg})

#     except Exception as e:
#         st.error(f"Error occurred: {e}")

# # Load article content from Wikipedia, text input, or URL scraping
# @st.cache_resource
# def load_article_content(input_type, input_text):
#     try:
#         if input_type == "Wikipedia Query":
#             loader = WikipediaLoader(query=input_text)
#             return loader.load()[0].page_content
#         elif input_type == "Text":
#             return input_text
#         elif input_type == "URL":
#             loader = WebBaseLoader(url=input_text)
#             return loader.load()[0].page_content
#     except Exception as e:
#         st.error(f"Failed to load content: {e}")
#         return None

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
#     if st.sidebar.button("Analyze"):
#         if not input_text:
#             st.error("Please provide valid input.")
#         else:
#             article_content = load_article_content(input_type, input_text)
#             if article_content:
#                 initialize_session_state()

#                 # Set up document retriever using the loaded article content
#                 st.session_state.retriever = setup_retriever(article_content)

#                 if not st.session_state.history:
#                     st.session_state.history.append({"role": "assistant", "content": "Hello! I am WATS. How can I assist you today?"})
#                     st.session_state.conversation_history = initialize_conversation()

#     # Handle chat input
#     chat_input = st.chat_input("Ask me a question about the article:")
#     if chat_input:
#         on_chat_submit(chat_input)

#     # Display chat history
#     for message in st.session_state.history:
#         role = message["role"]
#         avatar_image = "imgs/avatar_streamly.png" if role == "assistant" else "imgs/stuser.png"
#         with st.chat_message(role, avatar=avatar_image):
#             st.write(message["content"])







# import os

# from dotenv import load_dotenv
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import Chroma
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.retrievers import WikipediaRetriever
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# # Load environment variables from .env
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# llm = ChatOpenAI(
#                 openai_api_key=OPENROUTER_API_KEY,  # Ensure the key is passed correctly
#                 openai_api_base=OPENROUTER_BASE_URL,  # Ensure the base URL is correct
#                 model_name="mistralai/pixtral-12b:free"  # Adjust based on model availability
#             )

# def setup_retriever(input_type: str, input_text: None):
#     if input_type == "Wikipedia Query":
#         retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

#     elif input_type == "Text":
#         data = input_text

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#         all_splits = text_splitter.split_text(data)

#         embeddings = HuggingFaceInferenceAPIEmbeddings(
#                         api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
#                     )

#         vectorstore = Chroma.from_texts(texts=all_splits, embedding=embeddings)

#         retriever = vectorstore.as_retriever(k=4)

#     elif input_type == "URL":
#         loader = WebBaseLoader(input_text)
#         data = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#         all_splits = text_splitter.split_documents(data)

#         embeddings = HuggingFaceInferenceAPIEmbeddings(
#                         api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
#                     )

#         vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

#         retriever = vectorstore.as_retriever(k=4)

#     return retriever





# def setup_prompt(input_type: str):
#     if input_type == "Wikipedia Query":
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, just "
#             "reformulate it if needed and otherwise return it as is."
#             "Remember if the latest user question does not reference context,"
#             "return as is."
#         )

#         # Create a prompt template for contextualizing questions
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         # Answer question prompt
#         # This system prompt helps the AI understand that it should provide concise answers
#         # based on the retrieved context and indicates what to do if the answer is unknown
#         qa_system_prompt = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, 
#         answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
#         You don't need to return user question.

#         Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
#         justifies the answer and the LINK of article. Return a citation for every quote across all articles 
#         that justify the answer. 


#         Here are the Wikipedia articles:{context}

#         Answer:
#         Citations:
#         """

#         # Create a prompt template for answering questions
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#     elif input_type == "Text":
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, just "
#             "reformulate it if needed and otherwise return it as is."
#             "Remember if the latest user question does not reference context,"
#             "return as is."
#         )

#         # Create a prompt template for contextualizing questions
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         # Answer question prompt
#         # This system prompt helps the AI understand that it should provide concise answers
#         # based on the retrieved context and indicates what to do if the answer is unknown
#         qa_system_prompt = """You're a helpful AI assistant. Given a user question and some article snippets, 
#         answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
#         You don't need to return user question.

#         Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
#         justifies the answer and the LINK of article. Return a citation for every quote across all articles 
#         that justify the answer. 


#         Here are the articles:{context}

#         Answer:
#         Citations:
#         """

#         # Create a prompt template for answering questions
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#     elif input_type == "URL":
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question which can be understood "
#             "without the chat history. Do NOT answer the question, just "
#             "reformulate it if needed and otherwise return it as is."
#             "Remember if the latest user question does not reference context,"
#             "return as is."
#         )

#         # Create a prompt template for contextualizing questions
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         # Answer question prompt
#         # This system prompt helps the AI understand that it should provide concise answers
#         # based on the retrieved context and indicates what to do if the answer is unknown
#         qa_system_prompt = """You're a helpful AI assistant. Given a user question and some article snippets, 
#         answer the user question and provide citations. If none of the articles answer the question, just say you don't know.
#         You don't need to return user question.

#         Remember, you must return both an answer and citations. A citation consists of a VERBATIM QUOTE that 
#         justifies the answer and the LINK of article. Return a citation for every quote across all articles 
#         that justify the answer. 


#         Here are the articles:{context}

#         Answer:
#         Citations:
#         """

#         # Create a prompt template for answering questions
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#     return contextualize_q_prompt, qa_prompt


# def on_chat_submit(input_type, input_text, chat_input, chat_history):
#     retriever = setup_retriever(input_type, input_text)
#     contextualize_q_prompt, qa_prompt = setup_prompt(input_type)

#     # Create a history-aware retriever
#     # This uses the LLM to help reformulate the question based on chat history
#     history_aware_retriever = create_history_aware_retriever(
#         llm, retriever, contextualize_q_prompt
#     )

#     # Create a chain to combine documents for question answering
#     # `create_stuff_documents_chain` feeds all retrieved context into the LLM
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#     # Create a retrieval chain that combines the history-aware retriever and the question answering chain
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     # Process the user's query through the retrieval chain
#     result = rag_chain.invoke({"input": chat_input, "chat_history": chat_history})
    
#     return result


# # Function to simulate a continual chat
# def continual_chat():
#     print("Start chatting with the AI! Type 'exit' to end the conversation.")
#     chat_history = []  # Collect chat history here (a sequence of messages)
#     while True:
#         query = input("You: ")
#         if query.lower() == "exit":
#             break
#         # Process the user's query through the retrieval chain
#         input_text = """
#         Q&A with RAG
#         Overview
#         One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as Retrieval Augmented Generation, or RAG.

#         What is RAG?
#         RAG is a technique for augmenting LLM knowledge with additional data.

#         LLMs can reason about wide-ranging topics, but their knowledge is limited to the public data up to a specific point in time that they were trained on. If you want to build AI applications that can reason about private data or data introduced after a model's cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

#         LangChain has a number of components designed to help build Q&A applications, and RAG applications more generally.

#         Note: Here we focus on Q&A for unstructured data. Two RAG use cases which we cover elsewhere are:

#         Q&A over SQL data
#         Q&A over code (e.g., Python)
#         RAG Architecture
#         A typical RAG application has two main components:

#         Indexing: a pipeline for ingesting data from a source and indexing it. This usually happens offline.

#         Retrieval and generation: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

#         The most common full sequence from raw data to answer looks like:

#         Indexing
#         Load: First we need to load our data. This is done with DocumentLoaders.
#         Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won't fit in a model's finite context window.
#         Store: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a VectorStore and Embeddings model.
#         index_diagram

#         Retrieval and generation
#         Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
#         Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data
#         retrieval_diagram

#         Table of contents
#         Quickstart: We recommend starting here. Many of the following guides assume you fully understand the architecture shown in the Quickstart.
#         Returning sources: How to return the source documents used in a particular generation.
#         Streaming: How to stream final answers as well as intermediate steps.
#         Adding chat history: How to add chat history to a Q&A app.
#         Hybrid search: How to do hybrid search.
#         Per-user retrieval: How to do retrieval when each user has their own private data.
#         Using agents: How to use agents for Q&A.
#         Using local models: How to use local models for Q&A.

#         """
#         input_type, chat_input, chat_history = "Text", query, chat_history    # "Wikipedia Query", None   "https://python.langchain.com/v0.1/docs/use_cases/question_answering/" 
#         result = on_chat_submit(input_type, input_text, chat_input, chat_history)
#         # Display the AI's response
#         print(f"AI: {result['answer']}")
#         # Update the chat history
#         chat_history.append(HumanMessage(content=query))
#         chat_history.append(AIMessage(content=result["answer"]))


# # Main function to start the continual chat
# if __name__ == "__main__":
#     continual_chat()








# from langchain_core.messages import AIMessage, HumanMessage
# from typing import Dict

# from langchain_core.runnables import RunnablePassthrough
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableBranch
# from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from dotenv import find_dotenv, load_dotenv
# import os

# # Load environment variables
# load_dotenv(find_dotenv())

# # Retrieve API keys from .env
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# llm = ChatOpenAI(
#                 openai_api_key=OPENROUTER_API_KEY,  # Ensure the key is passed correctly
#                 openai_api_base=OPENROUTER_BASE_URL,  # Ensure the base URL is correct
#                 model_name="mistralai/pixtral-12b:free"  # Adjust based on model availability
#             )

# loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# embeddings = HuggingFaceInferenceAPIEmbeddings(
#                     api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
#                 )

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# # k is the number of chunks to retrieve
# retriever = vectorstore.as_retriever(k=4)

# query_transform_prompt = ChatPromptTemplate.from_messages(
#     [
#         MessagesPlaceholder(variable_name="messages"),
#         (
#             "user",
#             "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
#         ),
#     ]
# )

# query_transforming_retriever_chain = RunnableBranch(
#     (
#         lambda x: len(x.get("messages", [])) == 1,
#         # If only one message, then we just pass that message's content to retriever
#         (lambda x: x["messages"][-1].content) | retriever,
#     ),
#     # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
#     query_transform_prompt | llm | StrOutputParser() | retriever,
# ).with_config(run_name="chat_retriever_chain")


# SYSTEM_TEMPLATE = """
# Answer the user's questions based on the below context. 
# If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

# <context>
# {context}
# </context>
# """

# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )



# document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# conversational_retrieval_chain = RunnablePassthrough.assign(
#     context=query_transforming_retriever_chain,
# ).assign(
#     answer=document_chain,
# )


# response = conversational_retrieval_chain.invoke(
#     {
#         "messages": [
#             HumanMessage(content="Can LangSmith help test my LLM applications?"),
#             AIMessage(
#                 content="Yes, LangSmith can help test and evaluate your LLM applications. It allows you to quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs. Additionally, LangSmith can be used to monitor your application, log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise."
#             ),
#             HumanMessage(content="Tell me more!"),
#         ],
#     }
# ) 

# print(response)

# stream = conversational_retrieval_chain.stream(
#     {
#         "messages": [
#             HumanMessage(content="Can LangSmith help test my LLM applications?"),
#             AIMessage(
#                 content="Yes, LangSmith can help test and evaluate your LLM applications. It allows you to quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs. Additionally, LangSmith can be used to monitor your application, log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise."
#             ),
#             HumanMessage(content="Tell me more!"),
#         ],
#     }
# )

# for chunk in stream:
#     print(chunk)


# from langchain_community.retrievers import WikipediaRetriever

# retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

# xml_system = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, 
# answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

# Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that 
# justifies the answer, the ID of the quote article and the LINK of article. Return a citation for every quote across all articles 
# that justify the answer. Use the following format for your final output:

# <cited_answer>
#     <answer></answer>
#     <citations>
#         <citation><source_id></source_id><quote></quote><link></link></citation>
#         <citation><source_id></source_id><quote></quote><link></link></citation>
#         ...
#     </citations>
# </cited_answer>

# Here are the Wikipedia articles:{context}"""


# xml_prompt = ChatPromptTemplate.from_messages(
#     [("system", xml_system), ("human", "{input}")]
# )


# from langchain_core.output_parsers import XMLOutputParser
# from typing import List
# from langchain_core.documents import Document



# def format_docs_xml(docs: List[Document]) -> str:
#     formatted = []
#     for i, doc in enumerate(docs):
#         doc_str = f"""\
#     <source id=\"{i}\">
#         <title>{doc.metadata['title']}</title>
#         <article_snippet>{doc.page_content}</article_snippet>
#     </source>"""
#         formatted.append(doc_str)
#     return "\n\n<sources>" + "\n".join(formatted) + "</sources>"


# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs_xml(x["context"])))
#     | xml_prompt
#     | llm
#     | XMLOutputParser()
# )

# retrieve_docs = (lambda x: x["input"]) | retriever

# chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
#     answer=rag_chain_from_docs
# )


# result = chain.invoke({"input": "How fast are cheetahs?"})




# from langchain_community.retrievers import WikipediaRetriever

# retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)


# query_transform_prompt = ChatPromptTemplate.from_messages(
#     [
#         MessagesPlaceholder(variable_name="messages"),
#         (
#             "user",
#             "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
#         ),
#     ]
# )

# query_transforming_retriever_chain = RunnableBranch(
#     (
#         lambda x: len(x.get("messages", [])) == 1,
#         # If only one message, then we just pass that message's content to retriever
#         (lambda x: x["messages"][-1].content) | retriever,
#     ),
#     # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
#     query_transform_prompt | llm | StrOutputParser() | retriever,
# ).with_config(run_name="chat_retriever_chain")


# SYSTEM_TEMPLATE = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, 
# answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

# Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that 
# justifies the answer, the ID of the quote article and the LINK of article. Return a citation for every quote across all articles 
# that justify the answer. Use the following format for your final output:

# <cited_answer>
#     <answer></answer>
#     <citations>
#         <citation><source_id></source_id><quote></quote><link></link></citation>
#         <citation><source_id></source_id><quote></quote><link></link></citation>
#         ...
#     </citations>
# </cited_answer>

# Here are the Wikipedia articles:{context}"""

# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )


# from langchain_core.output_parsers import XMLOutputParser
# from typing import List
# from langchain_core.documents import Document



# def format_docs_xml(docs: List[Document]) -> str:
#     formatted = []
#     for i, doc in enumerate(docs):
#         doc_str = f"""\
#     <source id=\"{i}\">
#         <title>{doc.metadata['title']}</title>
#         <article_snippet>{doc.page_content}</article_snippet>
#         <link>{doc.metadata['source']}</link>
#     </source>"""
#         formatted.append(doc_str)
#     return "\n\n<sources>" + "\n".join(formatted) + "</sources>"


# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs_xml(x["context"])))
#     | question_answering_prompt
#     | llm
#     | XMLOutputParser()
# )

# retrieve_docs = (lambda x: x["messages"]) | retriever

# conversational_retrieval_chain = RunnablePassthrough.assign(
#     context=retrieve_docs,
# ).assign(
#     answer=rag_chain_from_docs,
# )




# result = conversational_retrieval_chain.invoke(
#     {
#         "messages": [
#             HumanMessage(content="How fast are cheetahs?"),
#             AIMessage(
#                 content="The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph)."
#             ),
#             HumanMessage(content="tell me more"),
#         ],
#     }
# ) 



# # print("\n##############################################\n")
# # print(result.keys())
# # print("\n##############################################\n")
# # print(result["messages"])
# # print("\n##############################################\n")
# # [print(a) for a in result["context"] ]
# print("\n##############################################\n")
# print(result["answer"])
# print("\n##############################################\n")
# print(result)