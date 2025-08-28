import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from huggingface_hub import InferenceClient

# -----------------------------
# PDF Text Loader
# -----------------------------
def load_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -----------------------------
# Chunk Text
# -----------------------------
def chunk_text(text, max_tokens=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], []
    current_len = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -----------------------------
# Simple Vector Store
# -----------------------------
class SimpleVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.metadata = []
        self.index = None

    def add(self, vectors, metas):
        for v, m in zip(vectors, metas):
            vec = np.array(v, dtype=np.float32)
            self.vectors.append(vec)
            self.metadata.append(m)
        if self.vectors:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.stack(self.vectors))

    def search(self, query_vector, k=5):
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query_vector, k)
        results = [self.metadata[i] for i in I[0]]
        return results

# -----------------------------
# Index PDF
# -----------------------------
def index_pdf(uploaded_file):
    text = load_pdf_text(uploaded_file)
    chunks = chunk_text(text)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks)
    store = SimpleVectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return embed_model, store, chunks

# -----------------------------
# Hugging Face Inference with InferenceClient
# -----------------------------
def query_hf_api(prompt, model_id="HuggingFaceH4/zephyr-7b-beta"):
    HF_TOKEN = os.getenv("HF_TOKEN")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    response = client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )
    return response

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Student Assisted Chatbot", page_icon="🤖", layout="wide")
st.title("🎓 Student Assisted Chatbot")
st.write("Upload your textbook (PDF) and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_input = st.text_input("Your question:")

if uploaded_file and user_input:
    try:
        embed_model, store, chunks = index_pdf(uploaded_file)

        query_vec = embed_model.encode([user_input])[0]
        relevant_chunks = store.search(query_vec, k=5)
        context = "\n".join(relevant_chunks)

        prompt = f"""
You are a helpful tutor. Based only on the context below, answer the question in complete sentences. 
If the context does not contain enough information, say "I could not find this in the text."

Context:
{context}

Question: {user_input}

Answer:
"""

        answer = query_hf_api(prompt)

        st.write("🧠 Answer")
        st.write(answer if answer else "Sorry, I couldn’t generate a complete answer.")
    except Exception as e:
        st.error(f"Error: {e}")
