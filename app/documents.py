"""Write operations on the document registry.

Shared by the JSON API (web UI) and the MCP write tools so both go through
exactly the same storage, ingest and cleanup path. Every function returns
(payload, HTTP status) - the routes hand that straight to JSONResponse, the
MCP tools serialise the payload and drop the status.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import get_number_of_pages  # noqa: E402

import jobs  # noqa: E402
import media  # noqa: E402
import search  # noqa: E402
import store  # noqa: E402
import textfiles  # noqa: E402

PROJECT_NAME_MAX = 100
DOCUMENT_NAME_MAX = 200

# Upper bound for anything entering the server, uploaded or downloaded. The
# container runs under a hard 2 GB memory cap while ingest workers already
# hold a PDF plus its tree in RAM, so an unbounded body would OOM it.
# Plain-text files are additionally capped at textfiles.MAX_TEXT_BYTES.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def save_upload(filename: str, data: bytes, project: str) -> tuple[dict, int]:
    """Store an uploaded file and queue PDF ingest.

    Re-uploading identical bytes into the same folder returns the document
    that is already there instead of a second copy.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        return {"error": f"file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}, 413

    digest = hashlib.sha256(data).hexdigest()
    for existing_id, entry in store.load_registry().items():
        if entry.get("sha256") == digest and entry.get("project", "") == project:
            return {"doc_id": existing_id, "status": entry.get("status", "done"), "duplicate": True}, 200

    doc_id = store.unique_doc_id(store.slugify(filename))

    # Anything that isn't a PDF is treated as plain text (code, notebooks,
    # markdown, ...): stored without LLM ingest, immediately "done".
    if not filename.lower().endswith(".pdf"):
        try:
            text = textfiles.convert_upload(filename, data)
        except ValueError as e:
            return {"error": str(e)}, 400
        os.makedirs(store.FILES_DIR, exist_ok=True)
        path = os.path.join(store.FILES_DIR, f"{doc_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        store.create_document(
            doc_id, doc_name=filename, project=project, pdf_path=path,
            page_count=None, doc_type="text", line_count=text.count("\n") + 1, status="done",
            sha256=digest,
        )
        return {"doc_id": doc_id, "status": "done"}, 201

    os.makedirs(store.PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(store.PDF_DIR, f"{doc_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(data)

    try:
        page_count = get_number_of_pages(pdf_path)
    except Exception:
        os.remove(pdf_path)
        return {"error": "file is not a readable PDF"}, 400

    store.create_document(
        doc_id, doc_name=filename, project=project, pdf_path=pdf_path, page_count=page_count, sha256=digest
    )
    jobs.enqueue(doc_id)
    return {"doc_id": doc_id, "status": "processing"}, 201


def delete(doc_id: str) -> tuple[dict, int]:
    entry = store.delete_document(doc_id)
    if not entry:
        return {"error": "not found"}, 404
    for path in (entry.get("pdf_path"), entry.get("tree_path"), entry.get("pages_path")):
        if path and os.path.isfile(path):
            os.remove(path)
    media.delete(doc_id)
    search.remove(doc_id)
    return {"deleted": doc_id}, 200


def move(doc_id: str, project: str) -> tuple[dict, int]:
    if not store.update_document(doc_id, project=project):
        return {"error": "not found"}, 404
    if project:
        store.add_project(project)
    return {"doc_id": doc_id, "project": project}, 200


def rename(doc_id: str, name: str) -> tuple[dict, int]:
    if not name or len(name) > DOCUMENT_NAME_MAX:
        return {"error": "invalid document name"}, 400
    if not store.update_document(doc_id, doc_name=name):
        return {"error": "not found"}, 404
    return {"doc_id": doc_id, "doc_name": name}, 200


def retry(doc_id: str) -> tuple[dict, int]:
    entry = store.load_registry().get(doc_id)
    if not entry:
        return {"error": "not found"}, 404
    if entry.get("type", "pdf") == "text":
        return {"error": "text documents are not ingested"}, 400
    if entry.get("status") in ("queued", "processing"):
        return {"error": "already queued or processing"}, 409
    if not os.path.isfile(entry.get("pdf_path", "")):
        return {"error": "PDF file is missing - delete and re-upload"}, 400
    store.update_document(doc_id, status="queued", error="")
    jobs.enqueue(doc_id)
    return {"doc_id": doc_id, "status": "queued"}, 200


def media_blocker(entry: dict) -> str:
    if not media.enabled():
        return "figure extraction is disabled (PAGEINDEX_FIGURE_MODEL=off)"
    if entry.get("type", "pdf") != "pdf":
        return "only PDFs have figures"
    if not os.path.isfile(entry.get("pdf_path", "")):
        return "PDF file is missing - delete and re-upload"
    if entry.get("media_status") in ("queued", "processing"):
        return "already queued or processing"
    return ""


def extract_figures(doc_id: str) -> tuple[dict, int]:
    entry = store.load_registry().get(doc_id)
    if not entry:
        return {"error": "not found"}, 404
    error = media_blocker(entry)
    if error:
        return {"error": error}, 409 if "already" in error else 400
    jobs.enqueue_media(doc_id)
    return {"doc_id": doc_id, "media_status": "queued"}, 200


def status(doc_id: str) -> tuple[dict, int]:
    entry = store.load_registry().get(doc_id)
    if not entry:
        return {"error": "not found"}, 404
    payload = {
        "doc_id": doc_id,
        "type": entry.get("type", "pdf"),
        "status": entry.get("status", "done"),
        "uploaded_at": entry.get("uploaded_at", ""),
        "indexed_at": entry.get("indexed_at", ""),
        "media_status": entry.get("media_status", ""),
    }
    if entry.get("error"):
        payload["error"] = entry["error"]
    if entry.get("media_error"):
        payload["media_error"] = entry["media_error"]
    if entry.get("figure_count") is not None:
        payload["figure_count"] = entry["figure_count"]
    progress = jobs.progress(doc_id)
    if progress:
        payload["progress"] = progress
    return payload, 200


def create_project(name: str) -> tuple[dict, int]:
    if not name or len(name) > PROJECT_NAME_MAX:
        return {"error": "invalid project name"}, 400
    store.add_project(name)
    return {"name": name}, 201


def rename_project(old: str, new: str) -> tuple[dict, int]:
    if not new or len(new) > PROJECT_NAME_MAX:
        return {"error": "invalid project name"}, 400
    if new != old and not store.rename_project(old, new):
        return {"error": "project not found or target name already exists"}, 400
    return {"name": new}, 200


def delete_project(name: str) -> tuple[dict, int]:
    if not store.remove_project(name):
        return {"error": "project not found or not empty"}, 400
    return {"deleted": name}, 200
