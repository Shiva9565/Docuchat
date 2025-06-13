import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Streamlit UI setup
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload content and chat with content")

# Groq API Key input
api_key = st.text_input("Enter the Groq API key:", type='password')

# Proceed only if API key is entered
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Chat session ID input
    session_id = st.text_input('Session ID', value='default_session')

    # Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # File uploader
    uploaded_files = st.file_uploader("Choose PDF file(s)", type='pdf', accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # Use tempfile for better file handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt to make questions history-aware
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without chat history. Do not answer the question. "
            "Just reformulate it if needed, otherwise return it as it is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}')
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Q&A chain prompt
        qa_system_prompt = (
            'You are an assistant for question-answering tasks. '
            'Use the following pieces of retrieved context to answer '
            'the question. If you don\'t know the answer, say you don\'t know. '
            'Use three sentences maximum and keep the answer concise.\n\n{context}'
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}')
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create the RAG chain directly without unnecessary wrapping
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to manage session history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

        # User input for question
        user_input = st.text_input('Your Question:')
        if user_input:
            try:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {'input': user_input},
                    config={'configurable': {'session_id': session_id}},
                )
                st.write('Assistant:', response['answer'])
                st.write('Chat History:', session_history.messages)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    else:
        st.info("Please upload at least one PDF file to begin.")
else:
    st.warning("Please enter your Groq API key to proceed.")