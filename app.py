# app_streamlit.py
import streamlit as st
import fitz  # PyMuPDF
import re
import os
import time
import json
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Embeddings & vector store imports (OpenAI or local)
try:
    import openai
except Exception:
    openai = None

from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# Utilities: Text extraction
# -------------------------
def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n", "", s)         # fix hyphenation
    s = re.sub(r"\n{3,}", "\n\n", s)  # compress newlines
    return s.strip()

def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        text = clean_text(text)
        pages.append({"page": i + 1, "text": text})
    return pages

# -------------------------
# Chunking (hybrid)
# -------------------------
def chunk_pages(pages: List[Dict[str,Any]], max_chars=3000, overlap_chars=300):
    """
    Produce chunks with page provenance.
    Returns list of {id, text, start_page, end_page}
    """
    chunks = []
    buffer = ""
    buffer_pages = set()
    cur_id = 0
    def flush():
        nonlocal buffer, buffer_pages, cur_id
        if buffer.strip():
            start_page = min(buffer_pages) if buffer_pages else None
            end_page = max(buffer_pages) if buffer_pages else None
            chunks.append({
                "id": f"chunk_{cur_id}",
                "text": buffer.strip(),
                "start_page": start_page,
                "end_page": end_page
            })
            cur_id += 1
    for p in pages:
        page_marker = f"[PAGE {p['page']}]"
        text = page_marker + "\n" + p["text"] + "\n\n"
        # if adding this page would exceed max_chars, flush current buffer first
        if len(buffer) + len(text) > max_chars:
            # if buffer is empty but page > max_chars, split page by paragraphs
            if not buffer:
                paras = re.split(r"\n\s*\n", text)
                for para in paras:
                    if len(para) > max_chars:
                        # force-slice
                        for i in range(0, len(para), max_chars - overlap_chars):
                            piece = para[i:i + max_chars]
                            buffer = piece
                            buffer_pages = {p['page']}
                            flush()
                            buffer = ""
                            buffer_pages = set()
                    else:
                        buffer = para
                        buffer_pages = {p['page']}
                        flush()
                        buffer = ""
                        buffer_pages = set()
                continue
            else:
                # flush existing buffer with overlap
                # create overlap tail
                tail = buffer[-overlap_chars:]
                flush()
                buffer = tail + text
                buffer_pages = {p['page']}
        else:
            buffer += text
            buffer_pages.add(p['page'])
    # final flush
    if buffer.strip():
        flush()
    return chunks

# -------------------------
# Embeddings backend
# -------------------------
class EmbeddingBackend:
    def __init__(self, mode="openai", openai_api_key=None, local_model_name="all-mpnet-base-v2"):
        self.mode = mode
        if self.mode == "openai":
            if openai is None:
                raise RuntimeError("openai package not available. Install openai or choose local mode.")
            openai.api_key = openai_api_key
        else:
            self.model = SentenceTransformer(local_model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.mode == "openai":
            # batch in safe sizes
            all_vecs = []
            batch = []
            for t in texts:
                batch.append(t)
                if len(batch) >= 16:
                    res = openai.Embedding.create(input=batch, model="text-embedding-3-small")
                    vecs = [d["embedding"] for d in res["data"]]
                    all_vecs.extend(vecs)
                    batch = []
            if batch:
                res = openai.Embedding.create(input=batch, model="text-embedding-3-small")
                vecs = [d["embedding"] for d in res["data"]]
                all_vecs.extend(vecs)
            arr = np.array(all_vecs).astype("float32")
            # normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            return arr / norms
        else:
            arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            arr = arr.astype("float32")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            return arr / norms

# -------------------------
# Simple FAISS wrapper
# -------------------------
class FaissIndex:
    def __init__(self, dim):
        # index for cosine similarity (vectors must be normalized)
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vecs: np.ndarray, metadatas: List[dict]):
        self.index.add(vecs)
        self.metadatas.extend(metadatas)

    def search(self, qvec: np.ndarray, k=5):
        D, I = self.index.search(np.array([qvec]), k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            out.append({"score": float(score), "metadata": self.metadatas[idx], "index": int(idx)})
        return out

# -------------------------
# Summarization using OpenAI LLM
# -------------------------
STRUCTURED_PROMPT = """
You are an expert academic summarizer. Given combined short summaries of many chunks from a single paper, produce valid JSON with fields:
- title (string)
- abstract (string, 3-6 sentences)
- methods (string, 3-6 sentences)
- results (list of short strings with numbers if present)
- insights (list of 3 short items)
- limitations (list of up to 3 short items)
- key_terms (list of keywords)
Return ONLY valid JSON.
Input:
<<CONTENT>>
{text}
<<END>>
"""

def per_chunk_summaries_openai(chunks_texts: List[str], openai_key, model="gpt-4o-mini"):
    openai.api_key = openai_key
    summaries = []
    for ch in chunks_texts:
        prompt = ("Summarize this paper chunk in 1-2 plain-language sentences. If numeric results "
                  "appear in the chunk, list them as 'RESULT: <text>'.\n\nCHUNK:\n" + ch[:2000])
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a concise scientific summarizer."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=200
        )
        summaries.append(resp["choices"][0]["message"]["content"].strip())
    return summaries

def final_structured_summary_openai(per_chunk_summaries: List[str], openai_key, model="gpt-4o-mini"):
    openai.api_key = openai_key
    combined = "\n\n".join(per_chunk_summaries[:60])  # limit length
    prompt = STRUCTURED_PROMPT.format(text=combined)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":"You produce JSON only."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=800
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    # try parse
    try:
        obj = json.loads(raw)
    except Exception:
        # try extract JSON substring
        m = re.search(r"\{(?:.|\n)*\}", raw)
        if m:
            obj = json.loads(m.group(0))
        else:
            obj = {"title":"", "abstract": raw, "methods":"", "results": [], "insights": [], "limitations": [], "key_terms": []}
    return obj

# -------------------------
# Claim verification
# -------------------------
def verify_claims(structured_summary: dict, emb_backend: EmbeddingBackend, faiss_idx: FaissIndex, chunks: List[dict], threshold=0.75):
    """
    For each sentence in abstract/methods/results/insights, check whether there's a supporting chunk.
    Returns summary with 'claim_checks' mapping claim->support list (score + pages)
    """
    claims = []
    for key in ["abstract", "methods"] + ["results"] + ["insights"]:
        val = structured_summary.get(key)
        if not val:
            continue
        if isinstance(val, list):
            items = val
        else:
            # split into sentences
            items = re.split(r'\.\s+', val)
            items = [it.strip() for it in items if it.strip()]
        for it in items:
            claims.append({"field": key, "text": it})
    # embed claims
    texts = [c["text"] for c in claims]
    if not texts:
        return {}
    qvecs = emb_backend.embed_texts(texts)
    checks = []
    for i, qv in enumerate(qvecs):
        hits = faiss_idx.search(qv, k=3)
        # filter by threshold
        supports = [{"score": h["score"], "start_page": h["metadata"].get("start_page"), "end_page": h["metadata"].get("end_page")} for h in hits if h["score"] >= threshold]
        checks.append({"claim": claims[i], "supports": supports, "top_hits": hits})
    return checks

# -------------------------
# Chat (retrieval + LLM)
# -------------------------
def answer_question_openai(question: str, emb_backend: EmbeddingBackend, faiss_idx: FaissIndex, openai_key: str, chunks: List[dict], k=5, model="gpt-4o-mini"):
    openai.api_key = openai_key
    qvec = emb_backend.embed_texts([question])[0]
    hits = faiss_idx.search(qvec, k=k)
    # build context from top hits
    context_texts = []
    for h in hits:
        md = h["metadata"]
        snippet = md["text"][:1500]
        context_texts.append(f"(pages {md.get('start_page')}-{md.get('end_page')}) {snippet}")
    context = "\n\n---\n\n".join(context_texts)
    system = "You are an assistant that answers questions about a single research paper. Use ONLY the provided context. If the answer is not present, reply 'Not found in the paper.'"
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely and include source page ranges for any factual statements."
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=512
    )
    ans = resp["choices"][0]["message"]["content"].strip()
    return ans, hits

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Research Paper Summarizer (LLM + GenAI)", layout="wide")
st.title("📚 AI Research Paper Summarizer — LLM + Generative AI")
st.markdown("Upload a PDF, generate a structured summary, verify claims, and chat with the paper. Supports OpenAI and local embedding models.")

# Sidebar controls
st.sidebar.header("Run settings")
mode = st.sidebar.selectbox("Mode", ["OpenAI (cloud: LLM + embe]()
