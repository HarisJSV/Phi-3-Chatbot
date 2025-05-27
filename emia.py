import os
import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st

# Session initialization
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

# Theme Toggle
theme_mode = st.sidebar.radio("Choose Theme:", ("Light", "Dark"), key="theme_mode")

# Apply custom CSS based on the theme
def set_theme():
    if st.session_state.theme_mode == "Dark":
        dark_mode_css = """
            <style>
                body {
                    background-color: #1e1e1e;
                    color: #f5f5f5;
                }
                .stButton>button {
                    background-color: #3c3c3c;
                    color: #f5f5f5;
                    border: 1px solid #555;
                }
                .stTextInput>div>div>input {
                    background-color: #3c3c3c;
                    color: white;
                }
                .stMarkdown {
                    color: #f5f5f5;
                }
            </style>
        """
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        light_mode_css = """
            <style>
                body {
                    background-color: white;
                    color: black;
                }
                .stButton>button {
                    background-color: #f0f0f0;
                    color: black;
                    border: 1px solid #ccc;
                }
                .stTextInput>div>div>input {
                    background-color: white;
                    color: black;
                }
                .stMarkdown {
                    color: black;
                }
            </style>
        """
        st.markdown(light_mode_css, unsafe_allow_html=True)

# Apply the selected theme
set_theme()

# Reset chat function
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# Display PDF
def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar for file upload and model selection
with st.sidebar:
    selected_model = st.selectbox(
        "Select your LLM:",
        ("Phi-3", "Llama-3"),
        index=0,
        key='selected_model'
    )

    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                st.session_state.file_cache.pop(file_key, None)

            # LLM selection
            if st.session_state.current_model == "Llama-3":
                llm = Ollama(model="llama3", request_timeout=120.0)
            elif st.session_state.current_model == "Phi-3":
                llm = Ollama(model="phi3", request_timeout=120.0)

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()

                    # Embedding model
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True
                    )
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Query engine setup
                    Settings.llm = llm
                    query_engine = index.as_query_engine(
                        streaming=True, similarity_top_k=1
                    )

                    # Custom prompt template
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, think step by step to answer the query in a crisp manner. "
                        "If you don't know the answer, say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Chat Section
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and response
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
