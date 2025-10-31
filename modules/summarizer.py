# modules/summarizer.py
import json
import re
from typing import List, Dict, Any, Optional

try:
    import openai
except Exception:
    openai = None

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
Input content:
<<CONTENT>>
{text}
<<END>>
"""

def per_chunk_summaries_openai(chunks: List[str], openai_api_key: str, model: str = "gpt-4o-mini"):
    """Produce 1–2 sentence micro-summaries per chunk using OpenAI Chat API."""
    if openai is None:
        raise RuntimeError("openai package not installed.")
    openai.api_key = openai_api_key
    summaries = []
    for ch in chunks:
        prompt = ("Summarize this paper chunk in 1-2 plain-language sentences. If numeric results "
                  "appear in the chunk, include them as short 'RESULT:' notes.\n\nCHUNK:\n" + ch[:2500])
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise scientific summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200
        )
        summaries.append(resp["choices"][0]["message"]["content"].strip())
    return summaries

def final_structured_summary_openai(per_chunk_summaries: List[str], openai_api_key: str, model: str = "gpt-4o-mini"):
    """Combine per-chunk summaries into a final structured JSON using OpenAI."""
    if openai is None:
        raise RuntimeError("openai package not installed.")
    openai.api_key = openai_api_key
    combined = "\n\n".join(per_chunk_summaries[:60])  # cap for cost & length
    prompt = STRUCTURED_PROMPT.format(text=combined)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    # try parse JSON
    try:
        obj = json.loads(raw)
    except Exception:
        # try to extract json substring
        m = re.search(r"\{(?:.|\n)*\}", raw)
        if m:
            obj = json.loads(m.group(0))
        else:
            obj = {"title": "", "abstract": raw, "methods": "", "results": [], "insights": [], "limitations": [], "key_terms": []}
    return obj

def summarize_with_openai(chunks_texts: List[str], openai_api_key: str, chunk_sample_limit: int = 60):
    """End-to-end OpenAI summarization (per-chunk micro + final synth)."""
    sample = chunks_texts[:chunk_sample_limit]
    per = per_chunk_summaries_openai(sample, openai_api_key)
    structured = final_structured_summary_openai(per, openai_api_key)
    return structured

def summarize_local_fallback(chunks_texts: List[str], title_hint: str = ""):
    """
    Simple fallback local summarizer: concatenates initial content and returns draft JSON.
    This is intentionally simple — replace with a local instruction model for better results.
    """
    combined = "\n\n".join([t[:1500] for t in chunks_texts[:30]])
    abstract = combined[:1000]
    methods = "Methods not generated: run with OpenAI or a local instruction model for structured output."
    return {
        "title": title_hint,
        "abstract": abstract,
        "methods": methods,
        "results": [],
        "insights": [],
        "limitations": [],
        "key_terms": []
    }
