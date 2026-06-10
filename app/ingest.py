#!/usr/bin/env python3
"""Ingest a PDF: build its PageIndex tree structure and register it.

Usage:
    python3 app/ingest.py /data/pdfs/lecture01.pdf
    python3 app/ingest.py /data/pdfs/lecture01.pdf --doc-id lecture01

Requires OPENAI_API_KEY (or another LiteLLM-supported key) in the environment.
This calls the LLM repeatedly to build the tree + summaries + description,
so it costs a small amount of API usage per document.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import page_index_main  # noqa: E402
from pageindex.utils import ConfigLoader  # noqa: E402

from store import DATA_DIR, PDF_DIR, TREE_DIR, load_registry, save_registry  # noqa: E402


def slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return slug or "document"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PageIndex tree for a PDF and register it")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--doc-id", default=None, help="Override the document id (default: derived from filename)")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    doc_id = args.doc_id or slugify(pdf_path)

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TREE_DIR, exist_ok=True)

    # Copy the PDF into the data dir if it isn't already there.
    target_pdf_path = os.path.join(PDF_DIR, f"{doc_id}.pdf")
    if os.path.abspath(pdf_path) != os.path.abspath(target_pdf_path):
        with open(pdf_path, "rb") as src, open(target_pdf_path, "wb") as dst:
            dst.write(src.read())

    print(f"Building PageIndex tree for {doc_id} ...")
    opt = ConfigLoader().load({"if_add_doc_description": "yes"})
    result = page_index_main(target_pdf_path, opt)

    tree_path = os.path.join(TREE_DIR, f"{doc_id}.json")
    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    registry = load_registry()
    registry[doc_id] = {
        "doc_name": result.get("doc_name", doc_id),
        "doc_description": result.get("doc_description", ""),
        "pdf_path": target_pdf_path,
        "tree_path": tree_path,
    }
    save_registry(registry)

    print(f"Done. doc_id={doc_id}")
    print(f"  doc_name: {registry[doc_id]['doc_name']}")
    print(f"  description: {registry[doc_id]['doc_description'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
