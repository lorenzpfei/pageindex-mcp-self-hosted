#!/usr/bin/env python3
"""Self-hosted PageIndex MCP server + minimal document management web UI.

MCP (streamable HTTP under /mcp): browse/retrieve tools backed by locally
generated tree structures. Tree-building uses an LLM (OpenAI by default) once
per document; serving these tools afterwards is free - the calling agent
(e.g. Claude) does the reasoning/navigation itself.

Web UI (under /): upload PDFs into projects, watch ingest status. The JSON API
lives under /api/. Everything except / and /health requires the bearer token.

Note: no `from __future__ import annotations` here - it turns annotations into
strings, which breaks FastMCP's tool signature inspection in mcp 1.9.x.
"""
import json
import os
import secrets
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from pageindex.retrieve import (  # noqa: E402
    _parse_pages,
    get_document as _get_document,
    get_document_structure as _get_document_structure,
    get_page_content as _get_page_content,
)
from pageindex.utils import get_number_of_pages  # noqa: E402

import jobs  # noqa: E402
import store  # noqa: E402
import textfiles  # noqa: E402

API_KEY = os.environ.get("PAGEINDEX_MCP_API_KEY")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Paths reachable without the bearer token. The index page contains no data;
# its API calls are all authenticated.
PUBLIC_PATHS = {"/", "/health", "/favicon.ico"}

# stateless + JSON responses: no in-memory sessions (survives redeploys without
# breaking connected clients) and no SSE streams to upset reverse proxies.
mcp = FastMCP(
    "pageindex-self-hosted",
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)


# ── MCP tools ────────────────────────────────────────────────────────────────

def _doc_info_or_error(doc_id: str):
    entry = store.load_registry().get(doc_id)
    if not entry:
        return None, json.dumps({"error": f"Document {doc_id} not found"})
    if entry.get("status") != "done":
        return None, json.dumps(
            {"error": f"Document {doc_id} is not ready (status: {entry.get('status')}, error: {entry.get('error', '')})"}
        )
    return store.load_doc_info(doc_id), None


@mcp.tool()
def list_documents() -> str:
    """List all documents with id, name, description, project (folder), type, size, and ingest status.

    type is "pdf" (sized in pages) or "text" (plain-text files like code,
    notebooks, markdown - sized in lines). Only documents with status "done"
    can be read with the other tools.
    """
    items = []
    for doc_id, entry in store.load_registry().items():
        item = {
            "doc_id": doc_id,
            "doc_name": entry.get("doc_name", ""),
            "doc_description": entry.get("doc_description", ""),
            "project": entry.get("project", ""),
            "type": entry.get("type", "pdf"),
            "status": entry.get("status", "done"),
        }
        if item["type"] == "text":
            item["line_count"] = entry.get("line_count")
        else:
            item["page_count"] = entry.get("page_count")
        items.append(item)
    return json.dumps(items, ensure_ascii=False)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Get metadata for a document: doc_id, doc_name, doc_description, type,
    and page_count (pdf) or line_count (text)."""
    doc_info, error = _doc_info_or_error(doc_id)
    if error:
        return error
    return _get_document(documents={doc_id: doc_info}, doc_id=doc_id)


@mcp.tool()
def get_document_structure(doc_id: str) -> str:
    """Get the hierarchical tree structure (titles, sections, summaries, page ranges) of a document.

    For PDFs over ~20 pages, call this first to find relevant sections, then
    use get_page_content() with targeted page ranges. Text documents have no
    tree - this returns a single section spanning all lines; just fetch the
    line ranges you need via get_page_content().
    """
    doc_info, error = _doc_info_or_error(doc_id)
    if error:
        return error
    if doc_info.get("type") == "text":
        return json.dumps(
            [{"title": doc_info.get("doc_name", doc_id), "start_line": 1, "end_line": doc_info.get("line_count")}]
        )
    return _get_document_structure(documents={doc_id: doc_info}, doc_id=doc_id)


@mcp.tool()
def get_page_content(doc_id: str, pages: str) -> str:
    """Get the text content of specific pages of a document.

    pages format: '5-7', '3,8', or '12'. For PDFs these are 1-indexed physical
    page numbers; for text documents they are 1-indexed line numbers (e.g.
    '1-200' for the first 200 lines).
    """
    doc_info, error = _doc_info_or_error(doc_id)
    if error:
        return error
    if doc_info.get("type") == "text":
        try:
            line_nums = _parse_pages(pages)
        except (ValueError, AttributeError) as e:
            return json.dumps({"error": f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})
        return textfiles.line_content(doc_info["path"], line_nums)
    return _get_page_content(documents={doc_id: doc_info}, doc_id=doc_id, pages=pages)


# ── Web UI / JSON API ────────────────────────────────────────────────────────

async def index(request):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def health(request):
    return PlainTextResponse("ok")


async def api_state(request):
    db = store.load_db()
    documents = [
        {
            "doc_id": doc_id,
            "doc_name": e.get("doc_name", ""),
            "project": e.get("project", ""),
            "type": e.get("type", "pdf"),
            "page_count": e.get("page_count"),
            "line_count": e.get("line_count"),
            "status": e.get("status", "done"),
            "error": e.get("error", ""),
            "uploaded_at": e.get("uploaded_at", ""),
        }
        for doc_id, e in db["documents"].items()
    ]
    return JSONResponse({"projects": db["projects"], "documents": documents})


async def api_create_project(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    name = (body.get("name") or "").strip()
    if not name or len(name) > 100:
        return JSONResponse({"error": "invalid project name"}, status_code=400)
    store.add_project(name)
    return JSONResponse({"name": name}, status_code=201)


async def api_rename_project(request):
    old = request.path_params["name"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    new = (body.get("name") or "").strip()
    if not new or len(new) > 100:
        return JSONResponse({"error": "invalid project name"}, status_code=400)
    if new != old and not store.rename_project(old, new):
        return JSONResponse({"error": "project not found or target name already exists"}, status_code=400)
    return JSONResponse({"name": new})


async def api_retry_document(request):
    doc_id = request.path_params["doc_id"]
    entry = store.load_registry().get(doc_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    if entry.get("type", "pdf") == "text":
        return JSONResponse({"error": "text documents are not ingested"}, status_code=400)
    if entry.get("status") == "processing":
        return JSONResponse({"error": "already processing"}, status_code=409)
    if not os.path.isfile(entry.get("pdf_path", "")):
        return JSONResponse({"error": "PDF file is missing - delete and re-upload"}, status_code=400)
    store.update_document(doc_id, status="processing", error="")
    jobs.enqueue(doc_id)
    return JSONResponse({"doc_id": doc_id, "status": "processing"})


async def api_delete_project(request):
    name = request.path_params["name"]
    if not store.remove_project(name):
        return JSONResponse({"error": "project not found or not empty"}, status_code=400)
    return JSONResponse({"deleted": name})


async def api_upload(request):
    form = await request.form()
    upload = form.get("file")
    project = (form.get("project") or "").strip()
    if upload is None or not getattr(upload, "filename", None):
        return JSONResponse({"error": "missing file"}, status_code=400)

    doc_id = store.unique_doc_id(store.slugify(upload.filename))

    # Anything that isn't a PDF is treated as plain text (code, notebooks,
    # markdown, ...): stored without LLM ingest, immediately "done".
    if not upload.filename.lower().endswith(".pdf"):
        raw = await upload.read(textfiles.MAX_TEXT_BYTES + 1)
        try:
            text = textfiles.convert_upload(upload.filename, raw)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        os.makedirs(store.FILES_DIR, exist_ok=True)
        path = os.path.join(store.FILES_DIR, f"{doc_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        store.create_document(
            doc_id, doc_name=upload.filename, project=project, pdf_path=path,
            page_count=None, doc_type="text", line_count=text.count("\n") + 1, status="done",
        )
        return JSONResponse({"doc_id": doc_id, "status": "done"}, status_code=201)

    os.makedirs(store.PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(store.PDF_DIR, f"{doc_id}.pdf")
    with open(pdf_path, "wb") as f:
        while chunk := await upload.read(1 << 20):
            f.write(chunk)

    try:
        page_count = get_number_of_pages(pdf_path)
    except Exception:
        os.remove(pdf_path)
        return JSONResponse({"error": "file is not a readable PDF"}, status_code=400)

    store.create_document(doc_id, doc_name=upload.filename, project=project, pdf_path=pdf_path, page_count=page_count)
    jobs.enqueue(doc_id)
    return JSONResponse({"doc_id": doc_id, "status": "processing"}, status_code=201)


async def api_delete_document(request):
    doc_id = request.path_params["doc_id"]
    entry = store.delete_document(doc_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    for path in (entry.get("pdf_path"), entry.get("tree_path"), entry.get("pages_path")):
        if path and os.path.isfile(path):
            os.remove(path)
    return JSONResponse({"deleted": doc_id})


# ── App assembly ─────────────────────────────────────────────────────────────

class BearerAuthMiddleware:
    """Pure ASGI middleware: BaseHTTPMiddleware buffers streamed responses,
    which can break the MCP streamable-HTTP transport."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            # The MCP handler is mounted at /mcp/; Starlette would otherwise
            # answer /mcp with a 307 redirect, and MCP clients drop the
            # Authorization header when following it.
            scope["path"] = "/mcp/"
        if scope["type"] == "http" and API_KEY and scope["path"] not in PUBLIC_PATHS:
            auth = ""
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    auth = value.decode("latin-1")
                    break
            if not secrets.compare_digest(auth, f"Bearer {API_KEY}"):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.extend(
        [
            Route("/", index),
            Route("/health", health),
            Route("/api/state", api_state),
            Route("/api/projects", api_create_project, methods=["POST"]),
            Route("/api/projects/{name}", api_rename_project, methods=["PATCH"]),
            Route("/api/projects/{name}", api_delete_project, methods=["DELETE"]),
            Route("/api/upload", api_upload, methods=["POST"]),
            Route("/api/documents/{doc_id}/retry", api_retry_document, methods=["POST"]),
            Route("/api/documents/{doc_id}", api_delete_document, methods=["DELETE"]),
        ]
    )
    jobs.start()
    return BearerAuthMiddleware(app)


app = build_app()


if __name__ == "__main__":
    import uvicorn

    if not API_KEY:
        print("WARNING: PAGEINDEX_MCP_API_KEY not set - server is unauthenticated!", file=sys.stderr)

    uvicorn.run(app, host=HOST, port=PORT)
