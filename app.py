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
    texts = []
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                texts.append((page_number, text))
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        full_text = "\n".join([p.text for p in doc.paragraphs])
        texts = [(1, full_text)]
    return texts

def index_documents():
    for fname in os.listdir(DOC_FOLDER):
        full_path = os.path.join(DOC_FOLDER, fname)
        text_blocks = extract_text(full_path)
        for page_num, text in text_blocks:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            vectors = model.encode(chunks).tolist()
            for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
                payload = {
                    "text": chunk,
                    "source": fname,
                    "page": page_num,
                    "chunk_id": idx + 1
                }
                point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                client.upsert(collection_name=COLLECTION_NAME, points=[point])

# Index on first run
if not client.count(COLLECTION_NAME).count:
    index_documents()

# Streamlit UI
st.set_page_config(page_title="Offline RAG Chatbot")
st.title("📄 Document-based Chatbot (Offline)")
query = st.text_input("🔍 Ask a question about your documents:")

if query:
    q_vector = model.encode([query])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=q_vector, limit=3)
    
    if not results:
        st.warning("❌ No relevant information found in the documents.")
    else:
        for r in results:
            st.markdown(f"**📂 Source:** `{r.payload['source']}`")
            st.markdown(f"**📄 Page:** `{r.payload['page']}` | **🧩 Chunk ID:** `{r.payload['chunk_id']}`")
            st.write(r.payload["text"])
            st.markdown("---")
