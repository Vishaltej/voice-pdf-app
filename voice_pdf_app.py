import os 
import faiss 
import numpy as np 
import pandas as pd 
from langchain.document_loaders import UnstructuredPDFLoader 
from sentence_transformers import SentenceTransformer 
import fitz 
import streamlit as st 
from io import BytesIO 
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline 
from pydub import AudioSegment 
from audiorecorder import audiorecorder 
import io 
import requests 

# Helper Functions 
def load_pdf(uploaded_file): 
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
    text = "" 
    metadata = [] 
    for page_num in range(doc.page_count): 
        page = doc.load_page(page_num) 
        text += page.get_text() 
        metadata.append({"page_number": page_num + 1, "page_text": page.get_text()})  
    return text, metadata 

def load_excel(file_path): 
    df = pd.read_excel(file_path) 
    return "\n".join([" ".join(map(str, row)) for row in df.values]) 

def split_into_chunks(text, max_length=512): 
    words = text.split() 
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)] 
    return list(set(chunks))  

model = SentenceTransformer('all-MiniLM-L6-v2') 

def embed_texts(texts): 
    embeddings = model.encode(texts) 
    return np.array(embeddings) 

embedding_dim = 384  
index = faiss.IndexFlatL2(embedding_dim)  
metadata_store = [] 
text_store = []  

def store_in_faiss(embeddings, metadata, texts): 
    if embeddings.shape[1] != embedding_dim: 
        raise ValueError(f"Embedding dimensionality mismatch: {embeddings.shape[1]} vs {embedding_dim}") 
    index.add(embeddings) 
    metadata_store.extend(metadata) 
    text_store.extend(texts)  

def retrieve(query, top_k=5): 
    query_embedding = embed_texts([query]) 
    distances, indices = index.search(query_embedding, top_k) 
    unique_results = set() 
    results = [] 
    for idx, i in enumerate(indices[0]): 
        if text_store[i] not in unique_results:  
            unique_results.add(text_store[i]) 
            results.append((text_store[i], metadata_store[i], distances[0][idx])) 
    return results 

st.title("Document Search and Retrieval with AI Assistant") 
st.write("Upload a PDF or Excel file and enter a query to retrieve similar documents.") 

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "xlsx"]) 
if uploaded_file: 
    if uploaded_file.type == "application/pdf": 
        st.write("Processing PDF...") 
        document_text, metadata = load_pdf(uploaded_file) 
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 
        st.write("Processing Excel...") 
        document_text = load_excel(uploaded_file) 
        metadata = [{} for _ in range(len(document_text))]  

    chunks = split_into_chunks(document_text) 
    st.write(f"Document split into {len(chunks)} chunks.") 

    st.write("Generating embeddings...") 
    embeddings = embed_texts(chunks) 

    store_in_faiss(embeddings, metadata, chunks)  
    st.write("Embeddings and metadata stored in FAISS.") 

    query = st.text_input("Enter your query:") 
    if query: 
        results = retrieve(query) 
        relevant_text = " ".join([result[0] for result in results]) 
        
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"  
        
        if MISTRAL_API_KEY: 
            try: 
                payload = { 
                    "model": "codestral-latest", 
                    "prompt": f"You are a helpful assistant. Answer the user's question based on the provided documents.\nUser: {query}\nDocuments: {relevant_text}\nAI:", 
                    "max_tokens": 150, 
                    "temperature": 0.7, 
                } 
                headers = { 
                    "Authorization": f"Bearer {MISTRAL_API_KEY}", 
                    "Content-Type": "application/json", 
                } 
                response = requests.post(MISTRAL_API_URL, json=payload, headers=headers) 
                if response.status_code == 200: 
                    data = response.json() 
                    if "choices" in data and len(data["choices"]) > 0: 
                        ai_response = data["choices"][0]["message"]["content"].strip() 
                        st.markdown("### AI Assistant:") 
                        st.markdown(f"##### {ai_response}") 
                    else: 
                        st.error("The API response format is invalid or missing 'choices'.") 
                else: 
                    st.error(f"Error {response.status_code}: {response.content.decode('utf-8', errors='ignore')}") 
            except Exception as e: 
                st.error(f"Error communicating with Mistral API: {e}") 
        else: 
            st.warning("Please set your MISTRAL_API_KEY as an environment variable!")