# 📄 Offline RAG Chatbot

A fully **offline** Retrieval-Augmented Generation (RAG) chatbot that answers user queries from PDF and DOCX documents **without any internet or API** — built with open-source LLM tools, Qdrant vector DB, and Streamlit UI.

---

## 🚀 Features

✅ Completely offline — no API, no LangChain, no internet  
📄 Supports `.pdf` and `.docx` documents  
🔍 Shows filename, page number, and chunk ID  
🧠 Semantic search using `all-MiniLM-L6-v2`  
📦 Uses local Qdrant as vector DB  
🖥️ Lightweight, works on 16GB GPU (e.g., Tesla T4)  
⚡ Single vector DB call per query  
❗ Shows "no result found" if answer not in context  

---

## 🛠️ Stack Used

| Component              | Tech Used                           |
|------------------------|-------------------------------------|
| Embedding Model        | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB              | `Qdrant` (local instance, no API)   |
| UI Framework           | `Streamlit`                         |
| Document Parsers       | `pdfplumber`, `python-docx`         |
| Language               | `Python`                            |

---

## 📁 Folder Structure

