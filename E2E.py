import os
import base64
import gc
import random
import tempfile
import time
import uuid
import numpy as np

from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import tensorflow_hub as hub
from rouge_score import rouge_scorer


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# Load the models for benchmarking
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Golden answers for predefined queries
golden_answers = {
    # Key represents the question, and value represents the corresponding golden answer
    "Where did Haris complete his undergraduate education?": 
    "Haris completed his B.E. in Computer Science and Engineering at St. Joseph's Institute Of Technology, Chennai, India.",

    "What programming languages is Haris proficient in?": 
    "Haris is proficient in Python, SQL, Java, and C.",

    "What was Haris' role during his internship at Larsen & Toubro?": 
    "During his internship at Larsen & Toubro Ltd., Haris automated data processing tasks, reducing manual data entry by 70% and enhancing data extraction accuracy by 20%.",
    
     "What project did Haris work on involving the MERN stack?": 
    "Haris developed an expense tracker using the MERN stack, integrating a responsive front-end with React and styled-components, increasing user retention by 30%.",

    "What certification does Haris hold related to cloud computing?": 
    "Haris holds the 'Microsoft Certified: Azure Data Science Associate' certification.",
    
    # Add more questions and their respective golden answers
}

# Utility functions for similarity and evaluation
def compute_cosine_similarity(embedding_1, embedding_2):
    return np.inner(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def evaluate_response(generated_answer, golden_answer):
    use_emb_generated = use_model([generated_answer])[0].numpy()
    use_emb_golden = use_model([golden_answer])[0].numpy()

    st_emb_generated = st_model.encode(generated_answer, convert_to_tensor=True)
    st_emb_golden = st_model.encode(golden_answer, convert_to_tensor=True)

    use_similarity = compute_cosine_similarity(use_emb_generated, use_emb_golden)
    st_similarity = util.pytorch_cos_sim(st_emb_generated, st_emb_golden).item()

    rouge_scores = scorer.score(golden_answer, generated_answer)
    
    return {
        "USE Cosine Similarity": use_similarity,
        "ST Cosine Similarity": st_similarity,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure
    }

with st.sidebar:

    selected_model = st.selectbox(
            "Select your LLM:",
            ("Phi-3", "Llama-3"),
            index=0,
            key='selected_model'  
        )

    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                st.session_state.file_cache.pop(file_key, None)

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
                        loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()

                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
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

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

    if prompt in golden_answers:
        golden_answer = golden_answers[prompt]
        
        evaluation_results = evaluate_response(full_response, golden_answer)
        
        st.markdown(f"### Evaluation Results for `{prompt}`")
        st.write(f"USE Cosine Similarity: {evaluation_results['USE Cosine Similarity']:.4f}")
        st.write(f"ST Cosine Similarity: {evaluation_results['ST Cosine Similarity']:.4f}")
        st.write(f"ROUGE-1: {evaluation_results['ROUGE-1']:.4f}")
        st.write(f"ROUGE-2: {evaluation_results['ROUGE-2']:.4f}")
        st.write(f"ROUGE-L: {evaluation_results['ROUGE-L']:.4f}")
