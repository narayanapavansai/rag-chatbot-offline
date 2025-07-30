import os
import uuid
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Constants
DOC_FOLDER = "data"
COLLECTION_NAME = "offline_docs"
CHUNK_SIZE = 500
EMBED_DIM = 384

# Load model and client
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Create collection if not exists
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

# Extract text
def extract_text(file_path):
    texts = []
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                texts.append((page_num, text))
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        full_text = "\n".join([p.text for p in doc.paragraphs])
        texts = [(1, full_text)]
    return texts

# Index documents
def index_documents():
    for fname in os.listdir(DOC_FOLDER):
        if not fname.lower().endswith(('.pdf', '.docx')):
            continue
        full_path = os.path.join(DOC_FOLDER, fname)
        text_blocks = extract_text(full_path)
        for page_num, text in text_blocks:
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            vectors = model.encode(chunks).tolist()
            for idx, (vector, chunk) in enumerate(zip(vectors, chunks), start=1):
                payload = {
                    "text": chunk,
                    "source": fname,
                    "page": page_num,
                    "chunk_id": idx
                }
                point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                client.upsert(collection_name=COLLECTION_NAME, points=[point])
                print(f"✅ Indexed {fname} | Page {page_num} | Chunk {idx}")

# Run
if __name__ == "__main__":
    index_documents()
    print("✅ Indexing complete.")
