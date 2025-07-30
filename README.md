# 🤖 Offline RAG Chatbot (PDF/DOCX Q&A using Local Vector DB)

An **offline Retrieval-Augmented Generation (RAG)** chatbot powered by **Qdrant**, **Transformers**, and **Streamlit**. This app lets you ask natural language questions from a collection of **PDF** and **DOCX** files **without internet or APIs**. It uses **sentence embeddings**, performs semantic search on a **local vector database**, and returns matching document chunks.

---

## 🚀 Features

- 💬 Ask questions across multiple documents
- 📄 Supports `.pdf` and `.docx`
- 🔍 Displays source filename, page number, and chunk ID
- ⚡ Fast local search using [Qdrant](https://qdrant.tech/)
- ✅ Fully offline — no HuggingFace or OpenAI APIs
- 🖥️ Simple Streamlit-based user interface

---

## 📁 Folder Structure

rag_chatbot/
├── app.py # Streamlit UI
├── rag_chatbot.py # PDF/DOCX parsing & indexing
├── requirements.txt # Python dependencies
├── README.md # This file
├── data/ # Raw input documents
│ ├── file1.pdf
│ ├── file2.docx
│ └── ...
└── qdrant_storage/ # Local Qdrant vector DB (auto-created)


---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # Windows
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Download & Start Qdrant (Local Vector DB)
Download the Qdrant Windows binary and extract it.

Then in CMD:

bash
Copy
Edit
cd path\to\qdrant-folder
qdrant.exe
This should run Qdrant on localhost:6333.

5. Index the Documents
Make sure your .pdf and .docx files are in the data/ folder. Then run:

bash
Copy
Edit
python rag_chatbot.py
This creates vector embeddings and stores them in Qdrant.

6. Launch the Chatbot UI
bash
Copy
Edit
streamlit run app.py
💡 Usage Guide
Ask a question in the text box.

The app retrieves top 3 relevant document chunks.

Each result includes:

✅ The matching content

📁 Source file name

📄 Page number

🧩 Chunk ID

If no results are found, it will display “❌ Not found”.

🧠 Architecture & Design
✅ Document Parsing
.pdf files parsed using pdfplumber

.docx files parsed using python-docx

✅ Chunking Strategy
Split each page’s text into logical chunks (~500 characters)

Each chunk is associated with:

filename

page number

chunk ID

raw text

✅ Embedding & Retrieval
Sentence embeddings generated using:

r
Copy
Edit
all-MiniLM-L6-v2 (384-dim)
Embedded vectors are stored in Qdrant, a fast local vector DB

During query, the user input is also embedded and matched using cosine similarity

⚙️ Hardware Used
Component	Spec
OS	Windows 11
CPU	Intel Core i5/i7
RAM	8 GB or higher
GPU	❌ Not required
Storage	100 MB (Qdrant + Docs)
Internet	❌ Not required

📌 Observations
Chunking by 500–800 characters balances context & precision

Qdrant’s local performance is near-instant (<100ms)

all-MiniLM-L6-v2 gives excellent results with low resource usage

Works fully offline without calling APIs

Easy to scale: Just add more files in /data

📝 Example Questions
You can try:

“What are the features of the Techline IO interface?”

“How to configure the 1756 module?”

“What does the LG document explain about safety?”
🧳 How to Submit
Zip the folder:

python
Copy
Edit
rag_chatbot.zip
Push to GitHub:

Include .gitignore

Include README.md

Include requirements.txt

Include /data folder with documents (or sample files)

🔗 Credits
Qdrant

Sentence Transformers

Streamlit

