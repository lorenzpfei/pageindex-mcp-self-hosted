"""Background ingest worker.

A single daemon thread processes uploads one at a time (tree building is
LLM- and memory-heavy; the server stays responsive while it runs).
"""
import json
import os
import queue
import sys
import threading
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import page_index_main  # noqa: E402
from pageindex.utils import ConfigLoader  # noqa: E402

import store  # noqa: E402

_queue: "queue.Queue[str]" = queue.Queue()
_started = False


def build_tree(doc_id: str) -> None:
    """Run PageIndex tree building for a registered document (blocking)."""
    entry = store.load_registry().get(doc_id)
    if not entry:
        return
    try:
        opt = ConfigLoader().load({"if_add_doc_description": "yes"})
        result = page_index_main(entry["pdf_path"], opt)
        tree_path = os.path.join(store.TREE_DIR, f"{doc_id}.json")
        os.makedirs(store.TREE_DIR, exist_ok=True)
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        store.update_document(
            doc_id,
            doc_description=result.get("doc_description", ""),
            tree_path=tree_path,
            status="done",
            error="",
        )
        print(f"Ingest done: {doc_id}")
    except Exception as e:
        store.update_document(doc_id, status="failed", error=f"{type(e).__name__}: {e}")
        print(f"Ingest failed: {doc_id}", file=sys.stderr)
        traceback.print_exc()


def enqueue(doc_id: str) -> None:
    _queue.put(doc_id)


def _worker() -> None:
    while True:
        doc_id = _queue.get()
        try:
            build_tree(doc_id)
        finally:
            _queue.task_done()


def recover_interrupted() -> None:
    """Mark documents stuck in "processing" (e.g. after a restart) as failed."""
    for doc_id, entry in store.load_registry().items():
        if entry.get("status") == "processing":
            store.update_document(
                doc_id, status="failed", error="Interrupted by server restart - delete and re-upload"
            )


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    recover_interrupted()
    threading.Thread(target=_worker, daemon=True, name="ingest-worker").start()
