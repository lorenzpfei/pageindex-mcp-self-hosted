# PageIndex MCP (self-hosted)

A self-hosted MCP server exposing [PageIndex](https://github.com/VectifyAI/PageIndex)'s
vectorless, reasoning-based document retrieval. The `pageindex/` directory is a
vendored copy of VectifyAI's open-source PageIndex package (MIT licensed, see
`pageindex/LICENSE.upstream`).

How it works:

- **Ingest** (`app/ingest.py`): builds a hierarchical "table of contents" tree
  for a PDF using an LLM (OpenAI by default, configurable via
  `pageindex/config.yaml` + LiteLLM). This costs a small amount of LLM usage,
  once per document.
- **Serve** (`app/server.py`): exposes `list_documents`, `get_document`,
  `get_document_structure`, `get_page_content` as MCP tools over streamable
  HTTP, protected by a bearer token. The connecting agent (e.g. Claude) does
  the navigation/reasoning itself - serving is free after ingest.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY` - used only during ingest (tree building)
   - `PAGEINDEX_MCP_API_KEY` - bearer token clients must send, e.g. `openssl rand -hex 32`

2. Build and start:

   ```bash
   docker compose up -d --build
   ```

3. Ingest a PDF (run inside the running container, or locally with the same deps):

   ```bash
   docker compose exec pageindex-mcp python3 app/ingest.py /data/pdfs/lecture01.pdf
   ```

   Or copy the PDF into `./data/pdfs/` first, then run ingest with that path.
   The tree structure is saved to `./data/trees/<doc_id>.json` and registered
   in `./data/documents.json`.

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

## Deployment

Expose port 8000 behind a reverse proxy (e.g. Dokploy/Traefik) with TLS and a
domain, then use that domain in the MCP client config above.

## Persistence

`./data/` (PDFs, generated trees, registry) is bind-mounted and persists
across rebuilds/restarts. Back it up if you don't want to re-run ingest.
