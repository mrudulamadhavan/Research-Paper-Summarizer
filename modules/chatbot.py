# modules/chatbot.py
from typing import List, Dict, Any, Optional

try:
    import openai
except Exception:
    openai = None

def answer_question_openai(question: str, emb_backend, faiss_store, openai_api_key: str, chunks: List[Dict[str, Any]], k: int = 5, model: str = "gpt-4o-mini"):
    """Retrieval + OpenAI answer. Returns answer string and hit metadata."""
    if openai is None:
        raise RuntimeError("openai package not installed.")
    openai.api_key = openai_api_key
    qvec = emb_backend.embed_texts([question])[0]
    hits = faiss_store.search(qvec, k=k)
    context_texts = []
    for h in hits:
        md = h["metadata"]
        snippet = md.get("text", "")[:1500]
        page_info = f"(pages {md.get('start_page')}-{md.get('end_page')})" if md.get("start_page") else ""
        context_texts.append(f"{page_info}\n{snippet}")
    context = "\n\n---\n\n".join(context_texts) if context_texts else "No context available."
    system = "You are an assistant that answers questions about a single research paper. Use ONLY the provided context. If the answer is not present, reply 'Not found in the paper.'"
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite page ranges where relevant."
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=512
    )
    ans = resp["choices"][0]["message"]["content"].strip()
    return ans, hits

def answer_question_fallback(question: str, emb_backend, faiss_store, chunks: List[Dict[str, Any]], k: int = 5):
    """Fallback answer: retrieve top snippet(s) and return concatenated excerpt (no LLM)."""
    qvec = emb_backend.embed_texts([question])[0]
    hits = faiss_store.search(qvec, k=k)
    if not hits:
        return "No relevant information found in the paper.", hits
    # Build short snippet concatenation with page info
    snippets = []
    for h in hits:
        md = h["metadata"]
        snippet = md.get("text", "")[:800].replace("\n", " ")
        page_info = f"(pages {md.get('start_page')}-{md.get('end_page')})" if md.get("start_page") else ""
        snippets.append(f"{page_info} {snippet}")
    answer = "Top relevant excerpts from the paper:\n\n" + "\n\n---\n\n".join(snippets)
    return answer, hits
