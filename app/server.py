#!/usr/bin/env python3
"""Self-hosted PageIndex MCP server.

Exposes the same browse/retrieve tools as PageIndex's hosted MCP, backed by
locally generated tree structures (see ingest.py). Tree-building uses an LLM
(OpenAI by default) once per document; serving these tools afterwards is free
- the calling agent (e.g. Claude) does the reasoning/navigation itself.

Note: no `from __future__ import annotations` here - it turns annotations into
strings, which breaks FastMCP's tool signature inspection in mcp 1.9.x.
"""
import json
import os
import secrets
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from pageindex.retrieve import (  # noqa: E402
    get_document as _get_document,
    get_document_structure as _get_document_structure,
    get_page_content as _get_page_content,
)
from store import load_doc_info, load_registry  # noqa: E402

API_KEY = os.environ.get("PAGEINDEX_MCP_API_KEY")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# stateless + JSON responses: no in-memory sessions (survives redeploys without
# breaking connected clients) and no SSE streams to upset reverse proxies.
mcp = FastMCP(
    "pageindex-self-hosted",
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
def list_documents() -> str:
    """List all ingested documents with their id, name, description, and page count."""
    registry = load_registry()
    items = [
        {
            "doc_id": doc_id,
            "doc_name": entry.get("doc_name", ""),
            "doc_description": entry.get("doc_description", ""),
            "page_count": entry.get("page_count"),
        }
        for doc_id, entry in registry.items()
    ]
    return json.dumps(items, ensure_ascii=False)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Get metadata for a document: doc_id, doc_name, doc_description, page_count."""
    doc_info = load_doc_info(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return _get_document(documents={doc_id: doc_info}, doc_id=doc_id)


@mcp.tool()
def get_document_structure(doc_id: str) -> str:
    """Get the hierarchical tree structure (titles, sections, summaries, page ranges) of a document.

    For documents over ~20 pages, call this first to find relevant sections,
    then use get_page_content() with targeted page ranges.
    """
    doc_info = load_doc_info(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return _get_document_structure(documents={doc_id: doc_info}, doc_id=doc_id)


@mcp.tool()
def get_page_content(doc_id: str, pages: str) -> str:
    """Get the text content of specific pages of a document.

    pages format: '5-7', '3,8', or '12' (1-indexed physical PDF page numbers).
    """
    doc_info = load_doc_info(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    return _get_page_content(documents={doc_id: doc_info}, doc_id=doc_id, pages=pages)


class BearerAuthMiddleware:
    """Pure ASGI middleware: BaseHTTPMiddleware buffers streamed responses,
    which can break the MCP streamable-HTTP transport."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and API_KEY and scope["path"] != "/health":
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


async def health(request):
    return PlainTextResponse("ok")


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.append(Route("/health", health))
    return BearerAuthMiddleware(app)


app = build_app()


if __name__ == "__main__":
    import uvicorn

    if not API_KEY:
        print("WARNING: PAGEINDEX_MCP_API_KEY not set - server is unauthenticated!", file=sys.stderr)

    uvicorn.run(app, host=HOST, port=PORT)
