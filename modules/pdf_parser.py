# modules/pdf_parser.py
import fitz  # PyMuPDF
import re

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n", "", s)         # fix hyphenation
    s = re.sub(r"\n{3,}", "\n\n", s)  # compress newlines
    s = s.strip()
    return s

def extract_text_from_pdf_bytes(pdf_bytes: bytes):
    """
    Returns list of pages: [{"page": int, "text": str}, ...]
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        try:
            raw = page.get_text("text")
        except Exception:
            raw = ""
        text = clean_text(raw)
        pages.append({"page": i + 1, "text": text})
    return pages
