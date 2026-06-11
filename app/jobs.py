"""Background ingest workers.

A small pool of daemon threads (PAGEINDEX_INGEST_WORKERS, default 2) builds
trees for uploaded PDFs. Each ingest holds the PDF + tree in memory
(~150-200 MB for typical lecture decks) and page_index_main already
parallelizes its LLM calls internally, so a high worker count mostly burns
RAM and OpenAI rate limits rather than speeding things up.
"""
import json
import os
import queue
import sys
import threading
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import page_index_main  # noqa: E402
from pageindex.utils import ConfigLoader, get_page_tokens  # noqa: E402

import ocr  # noqa: E402
import store  # noqa: E402

WORKERS = max(1, int(os.environ.get("PAGEINDEX_INGEST_WORKERS", "2")))
MODEL = os.environ.get("PAGEINDEX_MODEL", "")  # empty = pageindex/config.yaml default

_queue: "queue.Queue[str]" = queue.Queue()
_started = False


class IngestError(Exception):
    """Ingest failure with a message meant for the UI (no exception-type noise)."""


def _required_api_key(model: str) -> str | None:
    m = (model or "").removeprefix("litellm/")
    if m.startswith("gemini"):
        return "GEMINI_API_KEY"
    if m.startswith(("gpt-", "openai/", "o1", "o3", "o4")):
        return "OPENAI_API_KEY"
    if m.startswith(("claude", "anthropic/")):
        return "ANTHROPIC_API_KEY"
    return None


def _check_api_keys(tree_model: str) -> None:
    """Fail fast with a clear message instead of burning retries on auth errors."""
    models = [tree_model] + ([ocr.MODEL] if ocr.enabled() else [])
    missing = {k for m in models if (k := _required_api_key(m)) and not os.environ.get(k)}
    if missing:
        raise IngestError(f"{' and '.join(sorted(missing))} not set - add it to .env, redeploy, then retry")


def build_tree(doc_id: str) -> None:
    """Run PageIndex tree building for a registered document (blocking)."""
    entry = store.load_registry().get(doc_id)
    if not entry:
        return
    store.update_document(doc_id, status="processing", error="")
    try:
        overrides = {"if_add_doc_description": "yes"}
        if MODEL:
            overrides["model"] = MODEL
        opt = ConfigLoader().load(overrides)
        _check_api_keys(opt.model)

        page_list = get_page_tokens(entry["pdf_path"], model=opt.model)
        page_list, ocr_pages = ocr.augment_page_list(entry["pdf_path"], page_list, model=opt.model)
        if ocr_pages:
            print(f"OCR transcribed {ocr_pages} text-poor page(s) for {doc_id}")

        result = page_index_main(entry["pdf_path"], opt, page_list=page_list)
        os.makedirs(store.TREE_DIR, exist_ok=True)
        tree_path = os.path.join(store.TREE_DIR, f"{doc_id}.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Cache page texts when OCR changed them: get_page_content() would
        # otherwise re-extract the sparse original at query time.
        pages_file = os.path.join(store.TREE_DIR, f"{doc_id}.pages.json")
        pages_path = ""
        if ocr_pages:
            pages_path = pages_file
            with open(pages_file, "w", encoding="utf-8") as f:
                json.dump(
                    [{"page": i + 1, "content": text} for i, (text, _) in enumerate(page_list)],
                    f, ensure_ascii=False,
                )
        elif os.path.isfile(pages_file):  # stale cache from a previous ingest
            os.remove(pages_file)

        store.update_document(
            doc_id,
            doc_description=result.get("doc_description", ""),
            tree_path=tree_path,
            pages_path=pages_path,
            status="done",
            error="",
            indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        print(f"Ingest done: {doc_id}")
    except Exception as e:
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        store.update_document(doc_id, status="failed", error=msg)
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


def requeue_interrupted() -> None:
    """Re-enqueue documents left in "queued"/"processing" by a restart.

    The queue only lives in memory, so a crash (e.g. an OOM kill) would
    otherwise silently drop everything that was still waiting. The original
    PDFs are on disk - re-running the ingest is always safe.
    """
    for doc_id, entry in store.load_registry().items():
        if entry.get("status") in ("queued", "processing"):
            store.update_document(doc_id, status="queued", error="")
            _queue.put(doc_id)
            print(f"Re-queued after restart: {doc_id}")


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    requeue_interrupted()
    for i in range(WORKERS):
        threading.Thread(target=_worker, daemon=True, name=f"ingest-worker-{i}").start()
