# PageIndex MCP (self-hosted)

A self-hosted MCP server exposing [PageIndex](https://github.com/VectifyAI/PageIndex)'s
vectorless, reasoning-based document retrieval. The `pageindex/` directory is a
vendored copy of VectifyAI's open-source PageIndex package (MIT licensed, see
`pageindex/LICENSE.upstream`).

How it works:

- **Ingest** (`app/ingest.py`): builds a hierarchical "table of contents" tree
  for a PDF using an LLM (Gemini Flash by default, configurable via
  `PAGEINDEX_MODEL` / `pageindex/config.yaml` + LiteLLM). This costs a small
  amount of LLM usage, once per document.
- **Hybrid OCR** (`app/ocr.py`): pages whose embedded text layer is too sparse
  (scans, image-heavy slides) are rendered and transcribed by a vision model
  during ingest; born-digital text pages are read losslessly for free. The
  transcriptions are cached so retrieval serves them too. Disable with
  `PAGEINDEX_OCR_MODEL=off`.
- **Serve** (`app/server.py`): exposes `list_documents`, `get_document`,
  `get_document_structure`, `get_page_content` as MCP tools over streamable
  HTTP, protected by a bearer token. The connecting agent (e.g. Claude) does
  the navigation/reasoning itself - serving is free after ingest.
- **Text files**: anything that isn't a PDF (code, Jupyter notebooks,
  markdown, any UTF-8 file up to 10 MB) is stored as plain text without LLM
  ingest - instantly available, zero cost. The same MCP tools serve them,
  with 1-indexed line numbers taking the role of page numbers (e.g.
  `get_page_content(doc_id, "1-200")` returns the first 200 lines).
  Notebook outputs are stripped on upload; only markdown and code cells are
  kept.
- **Web UI** (`/`): minimal document manager - create folders (projects),
  upload PDFs and text files (PDF ingest runs in background workers), watch
  queued/processing status, rename and delete documents. Clicking a document
  opens a detail view (description, dates) with a button to open the original
  PDF/file in a new tab. Unlock with the same bearer token; it is kept in the
  browser's localStorage.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `GEMINI_API_KEY` - used only during ingest (tree building + OCR)
   - `PAGEINDEX_MCP_API_KEY` - bearer token clients must send, e.g. `openssl rand -hex 32`
   - optional: `PAGEINDEX_MODEL` (any LiteLLM model id for tree-building;
     default gemini/gemini-3.5-flash), `PAGEINDEX_OCR_MODEL` (vision model
     for text-poor pages, "off" to disable) and `PAGEINDEX_INGEST_WORKERS`
     (parallel PDF ingests, default 2)

2. Build and start:

   ```bash
   docker compose up -d --build
   ```

3. Upload PDFs and text files via the web UI at `https://<your-domain>/`
   (unlock with the `PAGEINDEX_MCP_API_KEY`). PDF ingest runs in the
   background; the list shows processing/done/failed per document. Text
   files are done immediately.

   Alternatively via CLI inside the container:

   ```bash
   docker compose exec pageindex-mcp python3 app/ingest.py /data/pdfs/lecture01.pdf --project "Machine Learning"
   ```

   Trees are saved to `<data>/trees/<doc_id>.json` and registered in
   `<data>/documents.json`.

## Connecting an MCP client

```json
{
  "mcpServers": {
    "pageindex-self": {
      "type": "http",
      "url": "https://<your-domain>/mcp",
      "headers": {
        "Authorization": "Bearer <PAGEINDEX_MCP_API_KEY>"
      }
    }
  }
}
```

For Claude Code:

```bash
claude mcp add --transport http pageindex-self https://<your-domain>/mcp \
  --header "Authorization: Bearer <PAGEINDEX_MCP_API_KEY>"
```

For opencode (`~/.config/opencode/opencode.json`, or a project-level
`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "pageindex-self": {
      "type": "remote",
      "url": "https://<your-domain>/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer <PAGEINDEX_MCP_API_KEY>"
      }
    }
  }
}
```

To keep the token out of the config file, opencode supports env substitution:
`"Authorization": "Bearer {env:PAGEINDEX_MCP_API_KEY}"`.

## Deployment

The compose file attaches the service to the external `dokploy-network`, so in
Dokploy you only need to add a domain pointing at service `pageindex-mcp`,
port `8000` (Traefik handles TLS). The container port is intentionally not
published on the host - the bearer token must only travel over HTTPS.

For plain local use (no Dokploy), swap the `networks` section for the
commented-out `127.0.0.1` port binding in `docker-compose.yml`.

`GET /health` is unauthenticated and returns `ok` - useful for uptime checks.

## Persistence

`../files/data/` (PDFs, generated trees, registry) is bind-mounted and persists
across rebuilds/restarts. On Dokploy this is the app's `files` storage dir,
which survives redeploys (the code dir does not). Back it up if you don't want
to re-run ingest.
