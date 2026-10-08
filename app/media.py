import base64
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import litellm
import pymupdf

from pageindex.utils import PROVIDER_BUSY_ERRORS, _llm_slots, completion_kwargs, litellm_model, retry_sleep

import store

MODEL = litellm_model(os.environ.get("PAGEINDEX_FIGURE_MODEL", "gemini/gemini-3.8-flash"))
MEDIA_DIR = os.path.join(store.DATA_DIR, "media")
PAGE_WIDTH = 1200
PAGE_QUALITY = 78
FIGURE_DPI = 200
FIGURE_QUALITY = 85
FIGURE_PADDING = 0.005
MIN_SIDE = 60
MAX_PARALLEL = 4
MAX_RETRIES = 8
MAX_ANSWER_RETRIES = 3
KINDS = ("diagram", "plot", "drawing", "photo", "table")
FIGURE_ID = re.compile(r"^p(\d+)-(\d+)$")

PROMPT = """This is one page of a university lecture (slides or lecture notes).
Find every figure a student would want to look at: diagrams, plots and charts, sketches and drawings, photos, screenshots, tables drawn as pictures, and graphical illustrations of formulas or algorithms.
Do not include plain text, bullet lists, formulas typeset as text, page headers or footers, slide numbers, university or course logos.
Answer with JSON only: {"figures": [{"box_2d": [ymin, xmin, ymax, xmax], "caption": "...", "kind": "..."}]}
- box_2d is normalised to 0-1000 and includes the figure's own labels, legend and axis titles, but not the surrounding paragraphs.
- caption is one sentence in the language of the page saying what the figure shows.
- kind is one of diagram, plot, drawing, photo, table, icon. Small symbols, logos and decorative images are icon.
If the page has no figures, answer {"figures": []}."""


def enabled() -> bool:
    return MODEL.strip().lower() not in ("", "off", "none", "no")


def doc_dir(doc_id: str) -> str:
    return os.path.join(MEDIA_DIR, doc_id)


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _render_page(pdf: pymupdf.Document, page_index: int) -> bytes:
    page = pdf[page_index]
    zoom = PAGE_WIDTH / page.rect.width
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pixmap.tobytes("jpeg", jpg_quality=PAGE_QUALITY)


def page_image(doc_id: str, pdf_path: str, page: int) -> bytes:
    path = os.path.join(doc_dir(doc_id), f"page-{page}.jpg")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    with pymupdf.open(pdf_path) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise ValueError(f"page {page} is out of range 1-{pdf.page_count}")
        data = _render_page(pdf, page - 1)
    os.makedirs(doc_dir(doc_id), exist_ok=True)
    _write_atomic(path, data)
    return data


def _parse_figures(text: str) -> list:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError(f"no JSON object in answer: {text[:200]!r}")
    figures = json.loads(match.group(0)).get("figures", [])
    if not isinstance(figures, list):
        raise ValueError("figures is not a list")
    return figures


def usable_figures(figures: list) -> list:
    usable = []
    for figure in figures:
        try:
            top, left, bottom, right = (min(1000.0, max(0.0, float(v))) for v in figure["box_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(figure.get("kind", "")).strip().lower()
        caption = str(figure.get("caption", "")).strip()
        if kind not in KINDS or not caption or bottom - top < MIN_SIDE or right - left < MIN_SIDE:
            continue
        usable.append({"kind": kind, "caption": caption, "box": (top, left, bottom, right)})
    return usable


def detect_figures(image: bytes) -> list | None:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
    ]}]
    for attempt in range(MAX_RETRIES):
        try:
            with _llm_slots:
                response = litellm.completion(
                    model=MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **completion_kwargs(MODEL),
                )
            return usable_figures(_parse_figures(response.choices[0].message.content))
        except PROVIDER_BUSY_ERRORS as e:
            logging.error(f"Figure detection hit a busy provider: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
        except Exception as e:
            logging.error(f"Figure detection failed: {e}")
            if attempt >= MAX_ANSWER_RETRIES - 1:
                return None
        time.sleep(retry_sleep(attempt))
    return None


def _crop(pdf: pymupdf.Document, page_index: int, box: tuple) -> bytes:
    page = pdf[page_index]
    width, height = page.rect.width, page.rect.height
    top, left, bottom, right = (v / 1000 for v in box)
    clip = pymupdf.Rect(
        (left - FIGURE_PADDING) * width,
        (top - FIGURE_PADDING) * height,
        (right + FIGURE_PADDING) * width,
        (bottom + FIGURE_PADDING) * height,
    ) & page.rect
    return page.get_pixmap(dpi=FIGURE_DPI, clip=clip, alpha=False).tobytes("jpeg", jpg_quality=FIGURE_QUALITY)


def _process_page(doc_id: str, pdf_path: str, page: int) -> list | None:
    directory = doc_dir(doc_id)
    with pymupdf.open(pdf_path) as pdf:
        image = _render_page(pdf, page - 1)
        _write_atomic(os.path.join(directory, f"page-{page}.jpg"), image)
        detected = detect_figures(image)
        if detected is None:
            return None
        figures = []
        for index, figure in enumerate(detected, start=1):
            file = f"figure-{page}-{index}.jpg"
            _write_atomic(os.path.join(directory, file), _crop(pdf, page - 1, figure["box"]))
            figures.append({
                "id": f"p{page}-{index}",
                "page": page,
                "index": index,
                "kind": figure["kind"],
                "caption": figure["caption"],
                "file": file,
            })
        return figures


def extract(doc_id: str, pdf_path: str, on_progress=None) -> tuple[int, list[int]]:
    directory = doc_dir(doc_id)
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)
    with pymupdf.open(pdf_path) as pdf:
        pages = list(range(1, pdf.page_count + 1))
    done, lock = 0, threading.Lock()

    def process(page: int) -> list | None:
        nonlocal done
        result = _process_page(doc_id, pdf_path, page)
        with lock:
            done += 1
            if on_progress:
                on_progress(done, len(pages))
        return result

    if on_progress:
        on_progress(0, len(pages))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        results = list(pool.map(process, pages))
    figures = [figure for result in results if result for figure in result]
    failed_pages = [page for page, result in zip(pages, results) if result is None]
    _write_atomic(
        os.path.join(directory, "figures.json"),
        json.dumps(figures, ensure_ascii=False, indent=1).encode(),
    )
    return len(figures), failed_pages


def load_figures(doc_id: str) -> list:
    path = os.path.join(doc_dir(doc_id), "figures.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def figure_image(doc_id: str, figure_id: str) -> bytes | None:
    match = FIGURE_ID.match(figure_id or "")
    if not match:
        return None
    path = os.path.join(doc_dir(doc_id), f"figure-{match.group(1)}-{match.group(2)}.jpg")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def delete(doc_id: str) -> None:
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)
