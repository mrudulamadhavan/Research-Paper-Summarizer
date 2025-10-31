# 
import re
from typing import List, Dict, Any

def chunk_pages(pages: List[Dict[str, Any]], max_chars: int = 3000, overlap_chars: int = 300):
    """
    Hybrid chunker that preserves page provenance.
    Returns list of {"id","text","start_page","end_page"}.
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
        buffer = ""
        buffer_pages.clear()

    for p in pages:
        page_marker = f"[PAGE {p['page']}]\n"
        text = page_marker + p.get("text", "") + "\n\n"
        # If adding the whole page exceeds, flush first
        if len(buffer) + len(text) > max_chars:
            if not buffer:
                # page itself is large -> split by paragraphs
                paras = re.split(r"\n\s*\n", text)
                for para in paras:
                    if not para.strip():
                        continue
                    if len(para) > max_chars:
                        # force-slice
                        for i in range(0, len(para), max_chars - overlap_chars):
                            piece = para[i:i + max_chars]
                            buffer = piece
                            buffer_pages.add(p["page"])
                            flush()
                    else:
                        buffer = para
                        buffer_pages.add(p["page"])
                        flush()
                continue
            else:
                # create small overlap tail
                tail = buffer[-overlap_chars:]
                flush()
                buffer = tail + "\n" + text
                buffer_pages.add(p["page"])
        else:
            buffer += text
            buffer_pages.add(p["page"])

    # final flush
    if buffer.strip():
        flush()
    return chunks
