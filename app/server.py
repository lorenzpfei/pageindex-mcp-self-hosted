#!/usr/bin/env python3
"""Self-hosted PageIndex MCP server + minimal document management web UI.

MCP (streamable HTTP under /mcp): browse/retrieve tools backed by locally
generated tree structures, plus write tools (upload/move/delete) that only
the full API key may call. Tree-building uses an LLM (OpenAI by default) once
per document; serving these tools afterwards is free - the calling agent
(e.g. Claude) does the reasoning/navigation itself.

Web UI (under /): upload PDFs into projects, watch ingest status. The JSON API
lives under /api/. Everything except / and /health requires the bearer token.

Note: no `from __future__ import annotations` here - it turns annotations into
strings, which breaks FastMCP's tool signature inspection in mcp 1.9.x.
"""
import asyncio
import base64
import contextvars
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import sys
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response  # noqa: E402
from starlette.routing import Route  # noqa: E402

from pageindex.retrieve import (  # noqa: E402
    _parse_pages,
    get_document as _get_document,
    get_document_structure as _get_document_structure,
    get_page_content as _get_page_content,
)

import documents  # noqa: E402
import jobs  # noqa: E402
import media  # noqa: E402
import search  # noqa: E402
import store  # noqa: E402
import textfiles  # noqa: E402

API_KEY = os.environ.get("PAGEINDEX_MCP_API_KEY")
# Derived so read-only clients need no extra secret in the deployment.
READ_API_KEY = hmac.new(API_KEY.encode(), b"pageindex-read-only", hashlib.sha256).hexdigest() if API_KEY else None
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Paths reachable without the bearer token. The index page contains no data;
# its API calls are all authenticated.
PUBLIC_PATHS = {"/", "/health", "/favicon.ico"}

# Set per request by BearerAuthMiddleware. The derived read-only key reaches
# /mcp (all read tools), so the write tools check this instead of the path.
_write_allowed = contextvars.ContextVar("pageindex_write_allowed", default=False)

MAX_REDIRECTS = 5
DOWNLOAD_TIMEOUT = 60.0

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
def search_pages(query: str, project: str = "", limit: int = 10) -> str:
    """Full-text search over the pages of all documents (BM25 over page text and section titles).

    Returns the best pages as [{doc_id, doc_name, project, page, section, snippet, score}].
    Use keywords in the document's language; slides are often English, regulations German,
    so search both when unsure. Words of four or more letters also match longer words
    starting with them. project optionally restricts the search to one folder.
    """
    return json.dumps(search.search(query, project, max(1, min(limit, 25))), ensure_ascii=False)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Get metadata for a document: doc_id, doc_name, doc_description, type,
    and page_count (pdf) or line_count (text)."""
    doc_info, error = _doc_info_or_error(doc_id)
    if error:
        return error
    return _get_document(documents={doc_id: doc_info}, doc_id=doc_id)


@mcp.tool()
def get_document_status(doc_id: str) -> str:
    """Check how far a document is: status ("queued", "processing", "done",
    "failed"), media_status for figure extraction, and the error text of
    whichever failed.

    While a worker is on it, "progress" carries phase ("reading", "ocr",
    "tree" during ingest, "figures" during figure extraction), done/total
    where the phase counts pages, and elapsed seconds. Poll this after
    upload_document() until status reads "done".
    """
    payload, _ = documents.status(doc_id)
    return json.dumps(payload, ensure_ascii=False)


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

    PDF pages list their extracted figures (id, kind, caption) under
    "figures"; fetch one with get_document_image(doc_id, id).
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
    content = _get_page_content(documents={doc_id: doc_info}, doc_id=doc_id, pages=pages)
    return _with_figures(doc_id, content)


def _with_figures(doc_id: str, content: str) -> str:
    figures = media.load_figures(doc_id)
    if not figures:
        return content
    try:
        page_items = json.loads(content)
    except ValueError:
        return content
    if not isinstance(page_items, list):
        return content
    by_page = {}
    for figure in figures:
        by_page.setdefault(figure["page"], []).append(
            {"id": figure["id"], "kind": figure["kind"], "caption": figure["caption"]}
        )
    for item in page_items:
        if isinstance(item, dict) and item.get("page") in by_page:
            item["figures"] = by_page[item["page"]]
    return json.dumps(page_items, ensure_ascii=False)


def _pdf_entry(doc_id: str) -> dict:
    entry = store.load_registry().get(doc_id)
    if not entry or entry.get("type", "pdf") != "pdf" or not os.path.isfile(entry.get("pdf_path", "")):
        raise ValueError(f"PDF document {doc_id} not found")
    return entry


@mcp.tool()
def list_figures(doc_id: str, pages: str = "") -> str:
    """List the figures (diagrams, plots, sketches, photos) extracted from a PDF.

    Returns [{id, page, kind, caption}]. pages optionally restricts the
    result, same format as get_page_content ('5-7', '3,8', '12'). Fetch a
    figure with get_document_image(doc_id, id). An empty list can also mean
    figure extraction has not run for this document yet.
    """
    _pdf_entry(doc_id)
    figures = media.load_figures(doc_id)
    if pages:
        try:
            wanted = set(_parse_pages(pages))
        except (ValueError, AttributeError) as e:
            return json.dumps({"error": f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})
        figures = [f for f in figures if f["page"] in wanted]
    return json.dumps(
        [{"id": f["id"], "page": f["page"], "kind": f["kind"], "caption": f["caption"]} for f in figures],
        ensure_ascii=False,
    )


@mcp.tool()
def get_document_image(doc_id: str, image_id: str) -> Image:
    """Get one extracted figure of a PDF as an image.

    image_id comes from list_figures() or the "figures" of get_page_content(),
    e.g. 'p12-1' (first figure on page 12).
    """
    _pdf_entry(doc_id)
    data = media.figure_image(doc_id, image_id)
    if data is None:
        raise ValueError(f"Image {image_id} not found in {doc_id}")
    return Image(data=data, format="jpeg")


@mcp.tool()
def get_page_image(doc_id: str, page: int) -> Image:
    """Get a rendered image of one PDF page (1-indexed physical page number).

    Use it when the layout matters or a figure was not extracted on its own.
    """
    entry = _pdf_entry(doc_id)
    return Image(data=media.page_image(doc_id, entry["pdf_path"], page), format="jpeg")


# ── MCP write tools (full API key only) ──────────────────────────────────────

def _require_write() -> None:
    if not _write_allowed.get():
        raise ValueError("This tool needs the full API key; the read-only key may only read.")


def _is_internal(ip) -> bool:
    """True for everything that is not a globally routable unicast address:
    loopback, private ranges, link-local (which covers the 169.254.169.254
    metadata endpoint), unique-local IPv6, multicast and reserved space.
    IPv4-mapped IPv6 is unwrapped first so ::ffff:127.0.0.1 cannot slip past."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped:
        ip = mapped
    return ip.is_multicast or not ip.is_global


async def _check_url(url: str) -> None:
    """Refuse anything but http(s) to a public address. Raises ValueError."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http and https URLs can be fetched, not {parsed.scheme or 'a relative URL'!r}")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {parsed.hostname}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if _is_internal(ip):
            raise ValueError(f"{parsed.hostname} resolves to {ip}, which is not a public address")


async def _download(url: str) -> tuple[str, bytes]:
    """Fetch a URL and return (final url, body).

    Redirects are followed by hand because every hop has to pass _check_url -
    httpx's own follow_redirects would only validate the first one.
    """
    async with httpx.AsyncClient(follow_redirects=False, timeout=DOWNLOAD_TIMEOUT) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _check_url(url)
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("redirect without a location header")
                    url = urllib.parse.urljoin(url, location)
                    continue
                response.raise_for_status()
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > documents.MAX_UPLOAD_BYTES:
                        raise ValueError(f"download exceeds {documents.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                return url, bytes(data)
    raise ValueError(f"more than {MAX_REDIRECTS} redirects")


@mcp.tool()
async def upload_document(url: str = "", content_base64: str = "", filename: str = "", project: str = "") -> str:
    """Add a document, either from a URL the server downloads or from base64 content.

    Pass exactly one of url (filename defaults to the last URL path segment)
    or content_base64 together with filename. project is the folder to file it
    under and is created if it does not exist yet. Only public http(s) URLs
    can be fetched, and no more than 100 MB.

    Re-uploading identical bytes into the same folder returns the document
    already there, flagged "duplicate": true, instead of a second copy.

    PDFs are queued for LLM tree-building and come back as status "processing";
    follow them with get_document_status(doc_id). Everything else is stored as
    plain text (UTF-8, max 10 MB) and is readable immediately.

    Needs the full API key.
    """
    _require_write()
    if bool(url) == bool(content_base64):
        return json.dumps({"error": "pass either url or content_base64"})
    if url:
        try:
            url, data = await _download(url)
        except Exception as e:
            return json.dumps({"error": f"download failed: {e}"})
        filename = filename or os.path.basename(urllib.parse.urlparse(url).path) or "download"
    else:
        if not filename:
            return json.dumps({"error": "filename is required with content_base64"})
        try:
            data = base64.b64decode(content_base64, validate=True)
        except Exception:
            return json.dumps({"error": "content_base64 is not valid base64"})
    payload, _ = documents.save_upload(filename, data, project.strip())
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def retry_document(doc_id: str) -> str:
    """Queue a failed PDF for ingest again. Needs the full API key."""
    _require_write()
    payload, _ = documents.retry(doc_id)
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def rename_document(doc_id: str, name: str) -> str:
    """Give a document a new display name. Needs the full API key."""
    _require_write()
    payload, _ = documents.rename(doc_id, name.strip())
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def move_document(doc_id: str, project: str = "") -> str:
    """Move a document into another folder; an empty project moves it to the top level.

    The folder is created if it does not exist yet. Needs the full API key.
    """
    _require_write()
    payload, _ = documents.move(doc_id, project.strip())
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def extract_figures(doc_id: str) -> str:
    """Queue figure extraction for a PDF: one vision-model call per page, so
    this costs LLM usage. Follow it with get_document_status(doc_id).

    Needs the full API key.
    """
    _require_write()
    payload, _ = documents.extract_figures(doc_id)
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def create_project(name: str) -> str:
    """Create a folder (project) that documents can be filed under.

    Needs the full API key.
    """
    _require_write()
    payload, _ = documents.create_project(name.strip())
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def rename_project(name: str, new_name: str) -> str:
    """Rename a folder and move every document in it along. Fails if the target
    name is already taken. Needs the full API key.
    """
    _require_write()
    payload, _ = documents.rename_project(name, new_name.strip())
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def delete_document(doc_id: str) -> str:
    """Delete a document and everything derived from it (tree, figures, search cache).

    This cannot be undone. Needs the full API key.
    """
    _require_write()
    payload, _ = documents.delete(doc_id)
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def delete_project(name: str) -> str:
    """Delete an empty folder; folders that still hold documents are refused.

    Needs the full API key.
    """
    _require_write()
    payload, _ = documents.delete_project(name)
    return json.dumps(payload, ensure_ascii=False)


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
            "doc_description": e.get("doc_description", ""),
            "project": e.get("project", ""),
            "type": e.get("type", "pdf"),
            "page_count": e.get("page_count"),
            "line_count": e.get("line_count"),
            "status": e.get("status", "done"),
            "error": e.get("error", ""),
            "uploaded_at": e.get("uploaded_at", ""),
            "indexed_at": e.get("indexed_at", ""),
            "media_status": e.get("media_status", ""),
            "media_error": e.get("media_error", ""),
            "figure_count": e.get("figure_count"),
            "progress": jobs.progress(doc_id),
        }
        for doc_id, e in db["documents"].items()
    ]
    return JSONResponse({"projects": db["projects"], "documents": documents})


async def api_create_project(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    payload, status = documents.create_project((body.get("name") or "").strip())
    return JSONResponse(payload, status_code=status)


async def api_rename_project(request):
    old = request.path_params["name"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    payload, status = documents.rename_project(old, (body.get("name") or "").strip())
    return JSONResponse(payload, status_code=status)


async def api_retry_document(request):
    payload, status = documents.retry(request.path_params["doc_id"])
    return JSONResponse(payload, status_code=status)


async def api_document_file(request):
    """Serve the stored original (PDF or text). The UI fetches it with the
    bearer header and opens it as a blob URL - a plain browser tab couldn't
    send the Authorization header itself."""
    doc_id = request.path_params["doc_id"]
    entry = store.load_registry().get(doc_id)
    if not entry or not os.path.isfile(entry.get("pdf_path", "")):
        return JSONResponse({"error": "not found"}, status_code=404)
    is_pdf = entry.get("type", "pdf") == "pdf"
    return FileResponse(
        entry["pdf_path"],
        media_type="application/pdf" if is_pdf else "text/plain",
        filename=entry.get("doc_name") or None,
        content_disposition_type="inline",
    )


async def api_extract_media(request):
    payload, status = documents.extract_figures(request.path_params["doc_id"])
    return JSONResponse(payload, status_code=status)


async def api_backfill_media(request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    project = (body.get("project") or "").strip()
    force = bool(body.get("force"))
    queued = []
    for doc_id, entry in store.load_registry().items():
        if project and entry.get("project") != project:
            continue
        if entry.get("status") != "done" or documents.media_blocker(entry):
            continue
        if entry.get("media_status") == "done" and not force:
            continue
        store.update_document(doc_id, media_attempts=0)
        jobs.enqueue_media(doc_id)
        queued.append(doc_id)
    return JSONResponse({"queued": queued})


async def api_page_image(request):
    doc_id = request.path_params["doc_id"]
    try:
        entry = _pdf_entry(doc_id)
        data = media.page_image(doc_id, entry["pdf_path"], int(request.path_params["page"]))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


async def api_figures(request):
    doc_id = request.path_params["doc_id"]
    if doc_id not in store.load_registry():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"figures": [
        {"id": f["id"], "page": f["page"], "kind": f["kind"], "caption": f["caption"]}
        for f in media.load_figures(doc_id)
    ]})


async def api_figure_image(request):
    doc_id = request.path_params["doc_id"]
    if doc_id not in store.load_registry():
        return JSONResponse({"error": "not found"}, status_code=404)
    data = media.figure_image(doc_id, request.path_params["figure_id"])
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


async def api_rename_document(request):
    doc_id = request.path_params["doc_id"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    payload, status = documents.rename(doc_id, (body.get("name") or "").strip())
    return JSONResponse(payload, status_code=status)


async def api_delete_project(request):
    payload, status = documents.delete_project(request.path_params["name"])
    return JSONResponse(payload, status_code=status)


async def api_upload(request):
    form = await request.form()
    upload = form.get("file")
    project = (form.get("project") or "").strip()
    if upload is None or not getattr(upload, "filename", None):
        return JSONResponse({"error": "missing file"}, status_code=400)
    # Bounded read: save_upload refuses anything over the limit, and the
    # extra byte is what makes an oversized body detectable without holding
    # all of it in memory.
    data = await upload.read(documents.MAX_UPLOAD_BYTES + 1)
    payload, status = documents.save_upload(upload.filename, data, project)
    return JSONResponse(payload, status_code=status)


async def api_delete_document(request):
    payload, status = documents.delete(request.path_params["doc_id"])
    return JSONResponse(payload, status_code=status)


# ── App assembly ─────────────────────────────────────────────────────────────

def _read_only_allowed(scope, auth: str) -> bool:
    if not secrets.compare_digest(auth, f"Bearer {READ_API_KEY}"):
        return False
    return scope.get("method") == "GET" or scope["path"].startswith("/mcp")


class BearerAuthMiddleware:
    """Pure ASGI middleware: BaseHTTPMiddleware buffers streamed responses,
    which can break the MCP streamable-HTTP transport."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            full_key = not API_KEY
            if API_KEY and scope["path"] not in PUBLIC_PATHS:
                auth = ""
                for name, value in scope.get("headers", []):
                    if name == b"authorization":
                        auth = value.decode("latin-1")
                        break
                full_key = secrets.compare_digest(auth, f"Bearer {API_KEY}")
                if not full_key and not _read_only_allowed(scope, auth):
                    response = JSONResponse({"error": "unauthorized"}, status_code=401)
                    await response(scope, receive, send)
                    return
            _write_allowed.set(full_key)
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
            Route("/api/documents/{doc_id}/file", api_document_file),
            Route("/api/documents/{doc_id}/media", api_extract_media, methods=["POST"]),
            Route("/api/documents/{doc_id}/pages/{page:int}/image", api_page_image),
            Route("/api/documents/{doc_id}/figures", api_figures),
            Route("/api/documents/{doc_id}/figures/{figure_id}/image", api_figure_image),
            Route("/api/media/backfill", api_backfill_media, methods=["POST"]),
            Route("/api/documents/{doc_id}", api_rename_document, methods=["PATCH"]),
            Route("/api/documents/{doc_id}", api_delete_document, methods=["DELETE"]),
        ]
    )
    jobs.start()
    search.start()
    return BearerAuthMiddleware(app)


app = build_app()


if __name__ == "__main__":
    import uvicorn

    if not API_KEY:
        print("WARNING: PAGEINDEX_MCP_API_KEY not set - server is unauthenticated!", file=sys.stderr)

    uvicorn.run(app, host=HOST, port=PORT)
