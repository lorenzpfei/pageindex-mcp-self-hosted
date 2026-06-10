#!/usr/bin/env python3
"""Self-hosted PageIndex MCP server.

Exposes the same browse/retrieve tools as PageIndex's hosted MCP, backed by
locally generated tree structures (see ingest.py). Tree-building uses an LLM
(OpenAI by default) once per document; serving these tools afterwards is free
- the calling agent (e.g. Claude) does the reasoning/navigation itself.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from pageindex.retrieve import (  # noqa: E402
    get_document as _get_document,
    get_document_structure as _get_document_structure,
    get_page_content as _get_page_content,
)
from store import load_doc_info, load_registry  # noqa: E402

API_KEY = os.environ.get("PAGEINDEX_MCP_API_KEY")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

mcp = FastMCP("pageindex-self-hosted", host=HOST, port=PORT)


@mcp.tool()
def list_documents() -> str:
    """List all ingested documents with their id, name, description, and page count."""
    registry = load_registry()
    items = []
    for doc_id, entry in registry.items():
        doc_info = load_doc_info(doc_id)
        page_count = json.loads(_get_document(documents={doc_id: doc_info}, doc_id=doc_id)).get("page_count")
        items.append(
            {
                "doc_id": doc_id,
                "doc_name": entry.get("doc_name", ""),
                "doc_description": entry.get("doc_description", ""),
                "page_count": page_count,
            }
        )
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


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if API_KEY:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {API_KEY}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app():
    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)
    return app


app = build_app()


if __name__ == "__main__":
    import uvicorn

    if not API_KEY:
        print("WARNING: PAGEINDEX_MCP_API_KEY not set - server is unauthenticated!", file=sys.stderr)

    uvicorn.run(app, host=HOST, port=PORT)
