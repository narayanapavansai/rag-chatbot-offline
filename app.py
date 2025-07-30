import os
import pdfplumber
import docx
import uuid
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Constants
DOC_FOLDER = "data"
COLLECTION_NAME = "offline_docs"

# Load model and Qdrant
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Create collection if not exists
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def index_documents():
    for fname in os.listdir(DOC_FOLDER):
        full_path = os.path.join(DOC_FOLDER, fname)
        text = extract_text(full_path)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        vectors = model.encode(chunks).tolist()
        payloads = [{"text": chunk, "source": fname} for chunk in chunks]
        points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p) for v, p in zip(vectors, payloads)]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

# Index on first run
if not client.count(COLLECTION_NAME).count:
    index_documents()

# Streamlit UI
st.set_page_config(page_title="📚 Offline RAG Chatbot")
st.title("📄 Document-based Chatbot (Offline)")
query = st.text_input("🔍 Ask a question about your documents:")

if query:
    q_vector = model.encode([query])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=q_vector, limit=3)
    for r in results:
        st.markdown(f"**📂 Source:** `{r.payload['source']}`")
        st.write(r.payload["text"])
        st.markdown("---")
