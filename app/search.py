"""Full-text search over the page texts and section titles of all documents.

Each document gets a cache file trees/<doc_id>.search.json with its page
texts and section titles, written after ingest (or by the warm-up thread for
documents ingested before search existed). Queries rank pages with BM25;
query terms of four or more characters also match longer words starting with
them, so "Masterarbeit" finds "Masterarbeiten" and "filter" finds "filtering".
"""

import bisect
import json
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter

import PyPDF2

import store
from pageindex.utils import strip_surrogates

CACHE_VERSION = 1
TEXT_BLOCK_LINES = 60
K1, B = 1.2, 0.75
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

_lock = threading.Lock()
_index: dict[str, dict] = {}


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _TOKEN.findall(folded) if len(t) > 1]


def _cache_path(doc_id: str) -> str:
    return os.path.join(store.TREE_DIR, f"{doc_id}.search.json")


def _section_titles(nodes, out):
    for node in nodes or []:
        start = node.get("start_index") or node.get("line_num")
        if start and node.get("title"):
            out.append({"page": int(start), "title": node["title"], "summary": node.get("summary", "")})
        _section_titles(node.get("nodes"), out)
    return out


def _page_texts(entry: dict) -> list[dict]:
    if entry.get("type") == "text":
        with open(entry["pdf_path"], "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        return [
            {"page": i + 1, "text": "\n".join(lines[i:i + TEXT_BLOCK_LINES])}
            for i in range(0, len(lines), TEXT_BLOCK_LINES)
        ]
    pages_path = entry.get("pages_path", "")
    if pages_path and os.path.isfile(pages_path):
        with open(pages_path, "r", encoding="utf-8") as f:
            return [{"page": p["page"], "text": p["content"]} for p in json.load(f)]
    with open(entry["pdf_path"], "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return [{"page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(reader.pages)]


def build(doc_id: str) -> None:
    """Write the search cache of one finished document."""
    entry = store.load_registry().get(doc_id)
    if not entry or entry.get("status") != "done":
        return
    sections = []
    if entry.get("tree_path") and os.path.isfile(entry["tree_path"]):
        with open(entry["tree_path"], "r", encoding="utf-8") as f:
            sections = _section_titles(json.load(f).get("structure", []), [])
    pages = [{**p, "text": strip_surrogates(p["text"])} for p in _page_texts(entry)]
    os.makedirs(store.TREE_DIR, exist_ok=True)
    tmp = _cache_path(doc_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "pages": pages, "sections": sections}, f, ensure_ascii=False)
    os.replace(tmp, _cache_path(doc_id))


def remove(doc_id: str) -> None:
    with _lock:
        _index.pop(doc_id, None)
    if os.path.isfile(_cache_path(doc_id)):
        os.remove(_cache_path(doc_id))


def _load(doc_id: str) -> dict | None:
    path = _cache_path(doc_id)
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _index.get(doc_id)
        if cached and cached["mtime"] == mtime:
            return cached
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != CACHE_VERSION:
        return None
    titles: dict[int, list[str]] = {}
    for section in data["sections"]:
        titles.setdefault(section["page"], []).append(section["title"])
    pages = []
    for page in data["pages"]:
        heading = " ".join(titles.get(page["page"], []))
        counts = Counter(_tokens(page["text"]))
        for token in _tokens(heading):
            counts[token] += 3
        pages.append({"page": page["page"], "text": page["text"], "heading": heading, "counts": counts, "length": sum(counts.values())})
    vocab = sorted({token for page in pages for token in page["counts"]})
    loaded = {"mtime": mtime, "pages": pages, "vocab": vocab}
    with _lock:
        _index[doc_id] = loaded
    return loaded


def _snippet(text: str, terms: list[str], width: int = 220) -> str:
    flat = " ".join(text.split())
    lower = flat.casefold()
    hit = next((i for i in (lower.find(t) for t in terms) if i >= 0), -1)
    start = max(0, hit - width // 3) if hit >= 0 else 0
    piece = flat[start:start + width]
    return ("…" if start > 0 else "") + piece + ("…" if start + width < len(flat) else "")


def _matching(vocab: list[str], term: str) -> list[str]:
    if len(term) < 4:
        i = bisect.bisect_left(vocab, term)
        return [term] if i < len(vocab) and vocab[i] == term else []
    out = []
    i = bisect.bisect_left(vocab, term)
    while i < len(vocab) and vocab[i].startswith(term):
        out.append(vocab[i])
        i += 1
    return out


def search(query: str, project: str = "", limit: int = 10) -> list[dict]:
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return []
    registry = store.load_registry()
    candidates = []
    for doc_id, entry in registry.items():
        if entry.get("status") != "done" or (project and entry.get("project", "").casefold() != project.casefold()):
            continue
        doc = _load(doc_id)
        if not doc:
            continue
        matches = {term: _matching(doc["vocab"], term) for term in terms}
        for page in doc["pages"]:
            counts = {term: sum(page["counts"].get(token, 0) for token in matches[term]) for term in terms}
            candidates.append((doc_id, page, counts))
    if not candidates:
        return []
    total = len(candidates)
    average = sum(page["length"] for _, page, _ in candidates) / total or 1
    frequencies = {term: sum(1 for _, _, counts in candidates if counts[term]) for term in terms}
    scored = []
    for doc_id, page, counts in candidates:
        score = 0.0
        for term in terms:
            tf = counts[term]
            if not tf:
                continue
            idf = math.log(1 + (total - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * page["length"] / average))
        if score > 0:
            scored.append((score, doc_id, page))
    scored.sort(key=lambda item: item[0], reverse=True)
    rare_first = sorted(terms, key=lambda term: frequencies[term])
    return [
        {
            "doc_id": doc_id,
            "doc_name": registry[doc_id].get("doc_name", ""),
            "project": registry[doc_id].get("project", ""),
            "page": page["page"],
            "section": page["heading"],
            "snippet": _snippet(page["text"], rare_first),
            "score": round(score, 2),
        }
        for score, doc_id, page in scored[:limit]
    ]


def _warm_up() -> None:
    for doc_id, entry in store.load_registry().items():
        if entry.get("status") == "done" and not os.path.isfile(_cache_path(doc_id)):
            try:
                build(doc_id)
            except Exception as e:
                print(f"Search cache failed for {doc_id}: {e}")
            time.sleep(0.2)


def start() -> None:
    threading.Thread(target=_warm_up, daemon=True, name="search-warm-up").start()
