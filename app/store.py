"""Document registry: tracks ingested PDFs and their PageIndex tree structures.

Layout under DATA_DIR:
  pdfs/<doc_id>.pdf       - original PDF
  trees/<doc_id>.json     - PageIndex tree structure (output of page_index_main)
  documents.json          - registry: {doc_id: {doc_name, doc_description, pdf_path, tree_path}}
"""
from __future__ import annotations

import json
import os

DATA_DIR = os.environ.get("PAGEINDEX_DATA_DIR", "/data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TREE_DIR = os.path.join(DATA_DIR, "trees")
REGISTRY_PATH = os.path.join(DATA_DIR, "documents.json")


def load_registry() -> dict:
    if not os.path.isfile(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def load_doc_info(doc_id: str) -> dict | None:
    """Build the doc_info dict expected by pageindex.retrieve functions."""
    registry = load_registry()
    entry = registry.get(doc_id)
    if not entry:
        return None
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
