#!/usr/bin/env python3
"""Ingest a PDF from the CLI: build its PageIndex tree structure and register it.

Usage:
    python3 app/ingest.py /data/pdfs/lecture01.pdf
    python3 app/ingest.py lecture01.pdf --project "Machine Learning" --doc-id lecture01

Requires OPENAI_API_KEY (or another LiteLLM-supported key) in the environment.
This calls the LLM repeatedly to build the tree + summaries + description,
so it costs a small amount of API usage per document.

(The web UI / POST /api/upload does the same thing via the background worker.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import get_number_of_pages  # noqa: E402

import jobs  # noqa: E402
import store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PageIndex tree for a PDF and register it")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--doc-id", default=None, help="Override the document id (default: derived from filename)")
    parser.add_argument("--project", default="", help="Project/folder name (created if missing)")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    doc_id = args.doc_id or store.unique_doc_id(store.slugify(pdf_path))
    os.makedirs(store.PDF_DIR, exist_ok=True)

    target_pdf_path = os.path.join(store.PDF_DIR, f"{doc_id}.pdf")
    if os.path.abspath(pdf_path) != os.path.abspath(target_pdf_path):
        with open(pdf_path, "rb") as src, open(target_pdf_path, "wb") as dst:
            dst.write(src.read())

    store.create_document(
        doc_id,
        doc_name=os.path.basename(pdf_path),
        project=args.project,
        pdf_path=target_pdf_path,
        page_count=get_number_of_pages(target_pdf_path),
    )

    print(f"Building PageIndex tree for {doc_id} ...")
    jobs.build_tree(doc_id)

    entry = store.load_registry()[doc_id]
    if entry["status"] != "done":
        print(f"FAILED: {entry.get('error')}", file=sys.stderr)
        return 1
    print(f"Done. doc_id={doc_id}")
    print(f"  doc_name: {entry['doc_name']}")
    print(f"  project: {entry['project'] or '(none)'}")
    print(f"  description: {entry['doc_description'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
