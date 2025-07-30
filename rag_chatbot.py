import os
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

DOC_FOLDER = "data"
COLLECTION_NAME = "offline_docs"
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

# Setup collection
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

# Index documents
for fname in os.listdir(DOC_FOLDER):
    full_path = os.path.join(DOC_FOLDER, fname)
    text = extract_text(full_path)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = model.encode(chunks).tolist()
    payloads = [{"text": chunk, "source": fname} for chunk in chunks]
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p) for v, p in zip(vectors, payloads)]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Documents indexed successfully.\n")

# Ask questions
while True:
    q = input("❓ Ask your question (or type 'exit'): ")
    if q.lower() == "exit":
        break
    q_vector = model.encode([q])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=q_vector, limit=3)
    for r in results:
        print(f"\n📄 Source: {r.payload['source']}\n{textwrap.fill(r.payload['text'], width=80)}\n{'-'*50}")
