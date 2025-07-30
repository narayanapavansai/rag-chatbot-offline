import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
import os

# Constants
COLLECTION_NAME = "offline_docs"
EMBED_DIM = 384

# Load model and Qdrant
st.set_page_config(page_title="Offline RAG Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            background-color: #fffbe6;
            border: 1px solid #f0ad4e;
            color: black !important;
        }
        .stButton button {
            background-color: #4CAF50;
            color: black;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Document-based Chatbot (Offline)")
st.caption("Ask questions from your local PDF & DOCX documents — powered by Qdrant & Transformers ✨")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Check if collection exists
if not client.collection_exists(COLLECTION_NAME):
    st.error("⚠️ Collection does not exist. Run `rag_chatbot.py` first to index documents.")
    st.stop()

# Input box
query = st.text_input("🔍 Ask a question about your documents:", placeholder="e.g. What does the Techline IO manual explain?")

# Search
if query:
    q_vector = model.encode([query])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=q_vector, limit=3)

    st.markdown("""<hr style='margin-top:1rem;margin-bottom:1rem'>""", unsafe_allow_html=True)

    if not results:
        st.warning("❌ Not found")
    else:
        for r in results:
            payload = r.payload
            with st.container():
                st.markdown(f"**📁 Source:** `{payload.get('source', '-')}`")
                st.markdown(f"**📄 Page:** `{payload.get('page', '-')}` | **🧩 Chunk ID:** `{payload.get('chunk_id', '-')}`")
                st.success(payload.get("text", ""))
                st.markdown("""<hr style='border:0.5px solid #ddd'>""", unsafe_allow_html=True)
