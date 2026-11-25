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
- **Figures** (`app/media.py`): after the tree is built, every page is
  rendered to a JPEG and a vision model (`PAGEINDEX_FIGURE_MODEL`, default
  gemini/gemini-4-hydrogen, one call per page) marks its figures - diagrams,
  plots, sketches, photos - with a bounding box and a caption. Each figure is
  cropped from the vector PDF at 200 DPI. Logos, icons and plain text are
  skipped. Documents ingested before this existed get their figures via the
  web UI ("Extract figures") or `POST /api/media/backfill`; their trees are
  left untouched. Disable with `PAGEINDEX_FIGURE_MODEL=off`.
- **Serve** (`app/server.py`): exposes `list_documents`, `get_document`,
  `get_document_structure`, `get_page_content`, `list_figures`,
  `get_document_image` and `get_page_image` as MCP tools over streamable
  HTTP, protected by a bearer token. `get_document_status` follows a
  document through ingest. Clients holding the full key additionally get the
  write tools (see below). `get_page_content` lists the figures of
  each returned page (id, kind, caption). `search_pages` ranks the pages of all
  documents by keyword (BM25 over page text and section titles, from a cache
  `trees/<doc_id>.search.json` written after ingest). The connecting agent
  (e.g. Claude) does the navigation/reasoning itself - serving is free after ingest.
- **Limits**: no upload or download may exceed 100 MB, and re-uploading
  identical bytes into the same folder returns the document already there
  (SHA-256 over the content) instead of a second copy.
- **Cost limits**: every ingest or figure-extraction run is an attempt, and
  a document gets at most 3 of each - container restarts count, so a PDF
  that crashes the container cannot loop. Each attempt is capped at 25 LLM
  calls per page (at least 200) and 3 hours; every request times out after
  10 minutes, and requests no retry can fix (bad key, rejected or oversized
  prompt) are not retried. A document that runs out ends up `failed` with
  the reason, until it is retried on purpose (web UI, `retry_document`,
  `extract_figures`), which grants a fresh budget. `get_document_status`
  shows the attempts used.
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
   - `PAGEINDEX_MCP_API_KEY` - bearer token clients must send, e.g. `openssl rand -hex 32`.
     Read-only clients (MCP tools and GET endpoints only, no upload or delete) can use the derived key
     `python3 -c 'import hashlib,hmac,sys; print(hmac.new(sys.argv[1].encode(), b"pageindex-read-only", hashlib.sha256).hexdigest())' <PAGEINDEX_MCP_API_KEY>`
   - optional: `PAGEINDEX_MODEL` (any LiteLLM model id for tree-building;
     default gemini/gemini-4-hydrogen), `PAGEINDEX_OCR_MODEL` (vision model
     for text-poor pages, "off" to disable), `PAGEINDEX_INGEST_WORKERS`
     (parallel PDF ingests, default 2) and `PAGEINDEX_MAX_CONCURRENT_LLM`
     (global cap on simultaneous LLM calls, default 8 - protects against
     provider rate limits; if an ingest still hits a rate limit or the
     model is overloaded, it is re-queued automatically with a growing
     cooldown, up to 3 attempts in total)

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

## MCP write tools

Clients holding the full `PAGEINDEX_MCP_API_KEY` can manage documents without
the web UI. The derived read-only key is rejected by all of them and keeps
working for everything else, `get_document_status` included.

- `upload_document(url, content_base64, filename, project)` - add a document,
  either from a URL the server downloads (`filename` defaults to the last URL
  path segment) or from base64 content plus `filename`. `project` is the
  folder; it is created when missing. PDFs come back as `"processing"` and are
  queued for tree-building - follow them with `get_document_status`. Anything
  else is stored as plain text and is readable immediately. Identical bytes in
  the same folder return the existing document with `"duplicate": true`.
- `retry_document(doc_id)` - queue a failed PDF for ingest again.
- `rename_document(doc_id, name)` - change a document's display name.
- `move_document(doc_id, project)` - move it to another folder; an empty
  `project` moves it to the top level.
- `extract_figures(doc_id)` - queue figure extraction (one vision-model call
  per page, so it costs LLM usage).
- `create_project(name)` / `rename_project(name, new_name)` - manage folders.
- `delete_document(doc_id)` - delete a document plus its tree, figures and
  search cache. Not undoable.
- `delete_project(name)` - delete an empty folder.

They share `app/documents.py` with the JSON API, so the UI and MCP clients
take the same storage, ingest and cleanup path.

### Fetching URLs safely

`upload_document(url=...)` makes the server issue a request of the caller's
choosing, so the URL is checked before each attempt:

- only `http` and `https`,
- the host is resolved first and **every** returned address must be a public
  unicast address - loopback, private ranges, link-local (which covers the
  `169.254.169.254` metadata endpoint), unique-local IPv6, multicast and
  reserved space are refused, with IPv4-mapped IPv6 unwrapped first,
- redirects are followed by hand, at most five, and each hop is checked again,
- the body is streamed and aborted past 100 MB; the request times out after
  60 s.

Residual risk: the address is validated before the request, not pinned for
it, so a DNS entry that changes between the check and the connection (DNS
rebinding) is not covered.

## Figure API

All endpoints need the bearer token.

- `GET /api/documents/<doc_id>/figures` - `{"figures": [{id, page, kind, caption}]}`
- `GET /api/documents/<doc_id>/figures/<id>/image` - figure crop (JPEG), e.g. id `p12-1`
- `GET /api/documents/<doc_id>/pages/<n>/image` - rendered page (JPEG, rendered on first request)
- `POST /api/documents/<doc_id>/media` - (re-)extract the figures of one document
- `POST /api/media/backfill` - extract figures for every finished PDF that has none yet;
  body `{"project": "Deep Learning"}` restricts it to a folder, `{"force": true}` redoes all

Files live under `<data>/media/<doc_id>/`.

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

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests
```

They run the real ASGI app in-process and cover the write path: that the
read-only key reaches no write tool and no writing REST endpoint, that the
tools do what they say (upload, rename, move, retry, folders, delete),
deduplication and the size cap, and that the URL fetch refuses non-public
targets and redirects into them.

## Persistence

`../files/data/` (PDFs, generated trees, registry) is bind-mounted and persists
across rebuilds/restarts. On Dokploy this is the app's `files` storage dir,
which survives redeploys (the code dir does not). Back it up if you don't want
to re-run ingest.
