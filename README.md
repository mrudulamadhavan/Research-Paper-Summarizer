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
