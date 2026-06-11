"""Hybrid vision OCR for text-poor PDF pages.

PyPDF2 reads the embedded text layer losslessly for born-digital PDFs, but
returns next to nothing for scans and image-heavy slides. Pages whose
extracted text falls below a threshold are rendered with pymupdf and
transcribed by a vision model instead - RAM-neutral (one page image at a
time, the API does the heavy lifting) and only the sparse pages cost money.

Failures degrade gracefully: a page that can't be transcribed keeps its
original extracted text.
"""
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import litellm
import pymupdf

from pageindex.utils import PROVIDER_BUSY_ERRORS, _llm_slots, completion_kwargs, count_tokens, retry_sleep

# Vision model for transcription (any LiteLLM model id); "off" disables OCR.
MODEL = os.environ.get("PAGEINDEX_OCR_MODEL", "gemini/gemini-3.5-flash")
# Pages with fewer extracted characters than this are sent to the vision model.
MIN_CHARS = int(os.environ.get("PAGEINDEX_OCR_MIN_CHARS", "100"))
DPI = 150
MAX_PARALLEL = 4
MAX_RETRIES = 8  # exponential backoff: must outlast 30s+ rate-limit windows

PROMPT = (
    "Transcribe this document page to plain text. Preserve headings and reading "
    "order, render tables as markdown, write formulas as LaTeX, and describe "
    "figures/diagrams briefly in square brackets. Return only the transcription; "
    "if the page is blank, return an empty string."
)


def enabled() -> bool:
    return MODEL.strip().lower() not in ("", "off", "none", "no")


def _render_page_png(pdf_path: str, page_index: int) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        return doc[page_index].get_pixmap(dpi=DPI).tobytes("png")


def _transcribe_page(pdf_path: str, page_index: int) -> str:
    """Transcribe one 0-indexed page; returns "" on failure."""
    image_b64 = base64.b64encode(_render_page_png(pdf_path, page_index)).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]}]
    for attempt in range(MAX_RETRIES):
        try:
            with _llm_slots:  # shared global cap, see pageindex.utils
                response = litellm.completion(model=MODEL, messages=messages, **completion_kwargs(MODEL))
            return response.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"OCR failed for page {page_index + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_sleep(attempt))
            elif isinstance(e, PROVIDER_BUSY_ERRORS):
                # Provider busy (out of quota or overloaded): abort the ingest
                # so jobs.py re-queues the document, instead of silently
                # indexing it with blank pages.
                raise
    return ""


def augment_page_list(pdf_path: str, page_list: list, model: str = None) -> tuple[list, int]:
    """Replace the text of text-poor pages with vision transcriptions.

    page_list is pageindex's [(page_text, token_count), ...], one tuple per
    physical page. Returns (new_page_list, number_of_pages_transcribed).
    """
    if not enabled():
        return page_list, 0
    sparse = [i for i, (text, _) in enumerate(page_list) if len((text or "").strip()) < MIN_CHARS]
    if not sparse:
        return page_list, 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        transcriptions = list(pool.map(lambda i: _transcribe_page(pdf_path, i), sparse))

    out = list(page_list)
    transcribed = 0
    for i, text in zip(sparse, transcriptions):
        text = (text or "").strip()
        if text:
            out[i] = (text, count_tokens(text, model))
            transcribed += 1
    return out, transcribed
