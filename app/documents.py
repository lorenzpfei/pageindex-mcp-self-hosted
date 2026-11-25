"""Write operations on the document registry.

Shared by the JSON API (web UI) and the MCP write tools so both go through
exactly the same storage, ingest and cleanup path.
"""
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


def save_upload(filename: str, data: bytes, project: str) -> tuple[dict, int]:
    """Store an uploaded file and queue PDF ingest. Returns (payload, HTTP status)."""
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

    store.create_document(doc_id, doc_name=filename, project=project, pdf_path=pdf_path, page_count=page_count)
    jobs.enqueue(doc_id)
    return {"doc_id": doc_id, "status": "processing"}, 201


def delete(doc_id: str) -> bool:
    entry = store.delete_document(doc_id)
    if not entry:
        return False
    for path in (entry.get("pdf_path"), entry.get("tree_path"), entry.get("pages_path")):
        if path and os.path.isfile(path):
            os.remove(path)
    media.delete(doc_id)
    search.remove(doc_id)
    return True


def move(doc_id: str, project: str) -> bool:
    if not store.update_document(doc_id, project=project):
        return False
    if project:
        store.add_project(project)
    return True


def create_project(name: str) -> str:
    """Create a folder. Returns an error message, empty when it worked."""
    if not name or len(name) > PROJECT_NAME_MAX:
        return "invalid project name"
    store.add_project(name)
    return ""
