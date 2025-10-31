# app.py
import streamlit as st
from modules.pdf_parser import extract_text_from_pdf_bytes
from modules.chunker import chunk_pages
from modules.embeddings import EmbeddingBackend
from modules.vectorstore import FaissStore
from modules import summarizer, chatbot
import numpy as np
import time

st.set_page_config(page_title="AI Research Paper Summarizer", layout="wide")
st.title("📄 AI Research Paper Summarizer — LLM + GenAI")

# Sidebar options
st.sidebar.header("Run settings")
mode = st.sidebar.selectbox("Mode", ["OpenAI (LLM + embeddings)", "Hybrid (OpenAI LLM + local embeddings)", "Local (no OpenAI)"])
openai_key = None
if "OpenAI" in mode:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
local_embed_model = st.sidebar.text_input("Local sentence-transformer model", value="all-mpnet-base-v2")
chunk_chars = st.sidebar.slider("Chunk max characters", 2000, 8000, 3000, step=500)
overlap_chars = st.sidebar.slider("Chunk overlap characters", 100, 1000, 300, step=50)
k_retrieval = st.sidebar.slider("Retriever top-k", 1, 8, 5)

uploaded = st.file_uploader("Upload research paper (PDF)", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

# Read PDF bytes
bytes_pdf = uploaded.read()
with st.spinner("Extracting pages..."):
    pages = extract_text_from_pdf_bytes(bytes_pdf)
st.success(f"Extracted {len(pages)} pages.")

# Chunk pages
with st.spinner("Chunking document..."):
    chunks = chunk_pages(pages, max_chars=chunk_chars, overlap_chars=overlap_chars)
st.write(f"Produced {len(chunks)} chunks.")

# Preview
if st.checkbox("Preview first chunk"):
    st.text_area("Chunk 0", chunks[0]["text"][:5000], height=300)

# Embedding backend
if mode == "OpenAI (LLM + embeddings)":
    if not openai_key:
        st.warning("Enter OpenAI API key in the sidebar.")
        st.stop()
    emb = EmbeddingBackend(mode="openai", openai_api_key=openai_key)
elif mode == "Hybrid (OpenAI LLM + local embeddings)":
    if not openai_key:
        st.warning("Enter OpenAI API key in the sidebar (LLM).")
        st.stop()
    emb = EmbeddingBackend(mode="local", local_model_name=local_embed_model)
else:
    emb = EmbeddingBackend(mode="local", local_model_name=local_embed_model)

# Build vectors and FAISS
with st.spinner("Computing embeddings and building vector index..."):
    texts = [c["text"] for c in chunks]
    vecs = emb.embed_texts(texts)
    dim = vecs.shape[1]
    store = FaissStore(dim)
    metadatas = [{"id": c["id"], "text": c["text"], "start_page": c["start_page"], "end_page": c["end_page"]} for c in chunks]
    store.add(vecs, metadatas)
st.success("Vector store is ready.")

# Summarization
st.header("Structured Summary")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Generate structured summary"):
        if "OpenAI" not in mode:
            st.warning("Structured synthesis uses OpenAI in this demo for best quality; local fallback will produce a rough output.")
            structured = summarizer.summarize_local_fallback(texts, title_hint=uploaded.name)
        else:
            with st.spinner("Generating per-chunk micro-summaries and final structured summary (OpenAI)..."):
                structured = summarizer.summarize_with_openai(texts, openai_api_key=openai_key)
        st.subheader("Structured Summary (JSON)")
        st.json(structured)
with col2:
    verify_threshold = st.slider("Claim verification threshold (cosine score)", 0.5, 0.95, 0.75, step=0.01)

if st.button("Verify claims (after summary generated)"):
    try:
        checks = []
        if "structured" not in locals():
            st.warning("Generate the structured summary first.")
        else:
            st.info("Verifying claims against indexed chunks...")
            checks = []
            # Use the verify_claims logic in this app (reimplemented inline to avoid circular imports)
            # We'll embed claims via emb and run search
            def extract_claims_from_struct(sdict):
                claims = []
                for key in ["abstract", "methods", "results", "insights"]:
                    val = sdict.get(key)
                    if not val:
                        continue
                    if isinstance(val, list):
                        items = val
                    else:
                        items = [c.strip() for c in re.split(r'\.\s+', val) if c.strip()]
                    for it in items:
                        claims.append({"field": key, "text": it})
                return claims
            import re
            claims = extract_claims_from_struct(structured)
            if not claims:
                st.write("No claims detected to verify.")
            else:
                claim_texts = [c["text"] for c in claims]
                qvecs = emb.embed_texts(claim_texts)
                for i, qv in enumerate(qvecs):
                    hits = store.search(qv, k=5)
                    supports = [ {"score": h["score"], "start_page": h["metadata"].get("start_page"), "end_page": h["metadata"].get("end_page")} for h in hits if h["score"] >= verify_threshold ]
                    checks.append({"claim": claims[i], "supports": supports, "top_hits": hits})
                st.subheader("Claim verification results")
                for c in checks:
                    st.markdown(f"**Claim ({c['claim']['field']}):** {c['claim']['text']}")
                    if c["supports"]:
                        for s in c["supports"]:
                            st.write(f"- supported (score={s['score']:.3f}) pages: {s['start_page']}-{s['end_page']}")
                    else:
                        if c["top_hits"]:
                            st.write("- No strong support above threshold. Top hits:")
                            for h in c["top_hits"]:
                                md = h["metadata"]
                                st.write(f"  - score {h['score']:.3f} pages {md.get('start_page')}-{md.get('end_page')}")
                        else:
                            st.write("- No retrieval hits found.")

# Chat interface
st.header("Chat with the paper")
question = st.text_input("Ask a question about this paper:")
if st.button("Ask question"):
    if not question:
        st.warning("Type a question.")
    else:
        if "OpenAI" in mode:
            ans, hits = chatbot.answer_question_openai(question, emb, store, openai_api_key=openai_key, chunks=chunks, k=k_retrieval)
            st.subheader("Answer")
            st.write(ans)
            st.subheader("Top source hits")
            for h in hits:
                md = h["metadata"]
                st.write(f"- score {h['score']:.3f} — pages {md.get('start_page')}-{md.get('end_page')}")
                st.text(md["text"][:500].replace("\n", " ") + ("..." if len(md["text"]) > 500 else ""))
        else:
            ans, hits = chatbot.answer_question_fallback(question, emb, store, chunks=chunks, k=k_retrieval)
            st.subheader("Answer (fallback retrieval snippets)")
            st.write(ans)
            st.subheader("Top source hits")
            for h in hits:
                md = h["metadata"]
                st.write(f"- score {h['score']:.3f} — pages {md.get('start_page')}-{md.get('end_page')}")
