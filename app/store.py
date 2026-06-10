"""Document registry: tracks projects, ingested PDFs, and plain-text files.

Layout under DATA_DIR:
  pdfs/<doc_id>.pdf       - original PDF
  files/<doc_id>.txt      - plain-text documents (stored as-is, no tree)
  trees/<doc_id>.json     - PageIndex tree structure (output of page_index_main)
  documents.json          - {"projects": [...], "documents": {doc_id: {...}}}

Document entry fields: type ("pdf" | "text", missing means "pdf"), doc_name,
doc_description, project, pdf_path (stored file path for all types), tree_path,
page_count, line_count (text only), status ("processing" | "done" | "failed"),
error, uploaded_at.
"""
import json
import os
import re
import threading
from datetime import datetime, timezone

DATA_DIR = os.environ.get("PAGEINDEX_DATA_DIR", "/data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
FILES_DIR = os.path.join(DATA_DIR, "files")
TREE_DIR = os.path.join(DATA_DIR, "trees")
REGISTRY_PATH = os.path.join(DATA_DIR, "documents.json")

_LOCK = threading.Lock()


def slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return slug or "document"


def _load_db() -> dict:
    if not os.path.isfile(REGISTRY_PATH):
        return {"projects": [], "documents": {}}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "documents" not in data:  # legacy flat {doc_id: entry} format
        data = {"projects": [], "documents": data}
        for entry in data["documents"].values():
            entry.setdefault("status", "done")
    return data


def _save_db(db: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def load_db() -> dict:
    with _LOCK:
        return _load_db()


def load_registry() -> dict:
    """All document entries: {doc_id: entry}."""
    return load_db()["documents"]


def add_project(name: str) -> bool:
    with _LOCK:
        db = _load_db()
        if name in db["projects"]:
            return False
        db["projects"].append(name)
        _save_db(db)
        return True


def remove_project(name: str) -> bool:
    """Remove a project. Fails (returns False) if any document still uses it."""
    with _LOCK:
        db = _load_db()
        if name not in db["projects"]:
            return False
        if any(e.get("project") == name for e in db["documents"].values()):
            return False
        db["projects"].remove(name)
        _save_db(db)
        return True


def rename_project(old: str, new: str) -> bool:
    """Rename a project and update all documents in it. Fails if the target
    name already exists (no implicit merging)."""
    with _LOCK:
        db = _load_db()
        if old not in db["projects"] or new in db["projects"]:
            return False
        db["projects"][db["projects"].index(old)] = new
        for entry in db["documents"].values():
            if entry.get("project") == old:
                entry["project"] = new
        _save_db(db)
        return True


def unique_doc_id(base: str) -> str:
    with _LOCK:
        docs = _load_db()["documents"]
        doc_id, n = base, 2
        while doc_id in docs:
            doc_id = f"{base}-{n}"
            n += 1
        return doc_id


def create_document(
    doc_id: str,
    doc_name: str,
    project: str,
    pdf_path: str,
    page_count: int | None,
    doc_type: str = "pdf",
    line_count: int | None = None,
    status: str = "processing",
) -> dict:
    entry = {
        "type": doc_type,
        "doc_name": doc_name,
        "doc_description": "",
        "project": project,
        "pdf_path": pdf_path,
        "tree_path": "",
        "page_count": page_count,
        "line_count": line_count,
        "status": status,
        "error": "",
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _LOCK:
        db = _load_db()
        if project and project not in db["projects"]:
            db["projects"].append(project)
        db["documents"][doc_id] = entry
        _save_db(db)
    return entry


def update_document(doc_id: str, **fields) -> dict | None:
    with _LOCK:
        db = _load_db()
        entry = db["documents"].get(doc_id)
        if not entry:
            return None
        entry.update(fields)
        _save_db(db)
        return entry


def delete_document(doc_id: str) -> dict | None:
    with _LOCK:
        db = _load_db()
        entry = db["documents"].pop(doc_id, None)
        if entry:
            _save_db(db)
        return entry


def load_doc_info(doc_id: str) -> dict | None:
    """Build the doc_info dict expected by pageindex.retrieve functions.

    Only valid for documents with status "done" (for PDFs, the tree file must
    exist; text documents have no tree).
    """
    entry = load_registry().get(doc_id)
    if not entry or entry.get("status") != "done":
        return None
    if entry.get("type", "pdf") == "text":
        return {
            "type": "text",
            "path": entry["pdf_path"],
            "doc_name": entry.get("doc_name", ""),
            "doc_description": entry.get("doc_description", ""),
            "line_count": entry.get("line_count"),
        }
    with open(entry["tree_path"], "r", encoding="utf-8") as f:
        tree = json.load(f)
    return {
        "type": "pdf",
        "path": entry["pdf_path"],
        "doc_name": entry.get("doc_name", tree.get("doc_name", "")),
        "doc_description": entry.get("doc_description", tree.get("doc_description", "")),
        "structure": tree.get("structure", []),
        "page_count": entry.get("page_count"),
    }
