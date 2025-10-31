# 🧠 AI Research Paper Summarizer

A smart web application that reads academic **research papers (PDFs)**, extracts their contents, and generates **human-friendly summaries, keywords, and insights** using **Large Language Models (LLMs)**. It also provides an **interactive chatbot** that allows users to ask questions about the paper, powered by **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

| Feature | Description |
|----------|--------------|
| 📄 **PDF Upload & Extraction** | Upload research papers (PDF format) and extract text automatically using PyMuPDF |
| ✂️ **Smart Chunking** | Breaks long texts into semantically meaningful sections for LLM processing |
| 🧠 **AI Summarization** | Uses GPT-based or open-source LLMs to generate structured, plain-language summaries |
| 💬 **Q&A Chatbot** | Chat with your paper using retrieval-augmented generation (RAG) |
| 🔍 **Keyword & Concept Extraction** | Automatically identifies important terms, models, datasets, and findings |
| ⚡ **Streamlit Frontend** | Clean, interactive UI built in Streamlit |
| 🧩 **Modular Codebase** | Clear separation of modules for easy extensibility |

---

## Folder Structure

<img width="771" height="489" alt="image" src="https://github.com/user-attachments/assets/446d8c64-f237-4b32-86e2-0dfa502fa5e6" />

## Tech Stack 

| Layer                       | Tools                                                      |
| --------------------------- | ---------------------------------------------------------- |
| **Frontend**                | Streamlit                                                  |
| **Backend / LLM Interface** | OpenAI API / Hugging Face Transformers                     |
| **Embeddings & Search**     | OpenAI Embeddings / Sentence Transformers + FAISS / Chroma |
| **Text Extraction**         | PyMuPDF (fitz)                                             |
| **Chunking & Workflow**     | LangChain                                                  |
| **Chat Handling**           | ConversationalRetrievalChain (LangChain)                   |
| **Language**                | Python 3.10+                                               |



## 🏗️ Project Architecture

```mermaid
graph TD
A[Upload PDF] --> B[Text Extraction via PyMuPDF]
B --> C[Text Chunking - LangChain Splitter]
C --> D[Embedding Generation - OpenAI/Sentence Transformers]
D --> E[Vector Store - FAISS/Chroma]
E --> F[Summarization - GPT-4/LLaMA]
E --> G[Q&A Retrieval Chain]
F --> H[Structured Summary Output]
G --> I[Chatbot Interface]


