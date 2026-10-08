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
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex import page_index_main  # noqa: E402
from pageindex.utils import PROVIDER_BUSY_ERRORS, ConfigLoader, get_page_tokens, litellm_model  # noqa: E402

import media  # noqa: E402
import ocr  # noqa: E402
import store  # noqa: E402

WORKERS = max(1, int(os.environ.get("PAGEINDEX_INGEST_WORKERS", "2")))
MODEL = litellm_model(os.environ.get("PAGEINDEX_MODEL", ""))  # empty = pageindex/config.yaml default

_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_started = False

_progress_lock = threading.Lock()
_progress: dict[str, dict] = {}


def _set_progress(doc_id: str, phase: str, done: int | None = None, total: int | None = None) -> None:
    with _progress_lock:
        previous = _progress.get(doc_id)
        started_at = previous["started_at"] if previous and previous["phase"] == phase else time.time()
        _progress[doc_id] = {"phase": phase, "done": done, "total": total, "started_at": started_at}


def _clear_progress(doc_id: str) -> None:
    with _progress_lock:
        _progress.pop(doc_id, None)


def progress(doc_id: str) -> dict | None:
    with _progress_lock:
        current = _progress.get(doc_id)
        if not current:
            return None
        return {
            "phase": current["phase"],
            "done": current["done"],
            "total": current["total"],
            "elapsed": round(time.time() - current["started_at"]),
        }

# Provider-busy cooldown: when the provider still answers 429 (quota) or 503
# (overloaded) after the per-call exponential backoff in pageindex.utils, it
# will stay that way for a while. Failing doc after doc would only burn more
# quota, so the document goes back to "queued" and all workers hold off,
# doubling the pause on every consecutive busy ingest (reset on success).
_rl_lock = threading.Lock()
_rl_until = 0.0
_rl_strikes = 0
RL_COOLDOWN_MAX = 1800  # 30 min


def _rate_limit_cooldown(kind: str, doc_id: str, e: Exception) -> None:
    global _rl_until, _rl_strikes
    with _rl_lock:
        _rl_strikes += 1
        cooldown = min(60 * 2 ** _rl_strikes, RL_COOLDOWN_MAX)
        _rl_until = max(_rl_until, time.time() + cooldown)
    message = f"LLM provider busy (rate limit/overload) - retrying automatically in ~{cooldown // 60} min ({e})"[:500]
    if kind == "media":
        store.update_document(doc_id, media_status="queued", media_error=message)
    else:
        store.update_document(doc_id, status="queued", error=message)
    _queue.put((kind, doc_id))
    print(f"Rate limited during {doc_id}; re-queued, pausing ingest {cooldown}s", file=sys.stderr)


def _rate_limit_wait() -> None:
    """Block the worker until the current cooldown window (if any) is over."""
    while True:
        wait = _rl_until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 30))


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


def _check_api_keys(*models: str) -> None:
    """Fail fast with a clear message instead of burning retries on auth errors."""
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
        _check_api_keys(opt.model, *([ocr.MODEL] if ocr.enabled() else []))

        _set_progress(doc_id, "reading")
        page_list = get_page_tokens(entry["pdf_path"], model=opt.model)
        page_list, ocr_pages = ocr.augment_page_list(
            entry["pdf_path"], page_list, model=opt.model,
            on_progress=lambda done, total: _set_progress(doc_id, "ocr", done, total),
        )
        _set_progress(doc_id, "tree")
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
        _reset_rate_limit_strikes()
        if media.enabled():
            enqueue_media(doc_id)
    except Exception as e:
        if _is_provider_busy(e):
            _rate_limit_cooldown("tree", doc_id, e)
            return
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        store.update_document(doc_id, status="failed", error=msg)
        print(f"Ingest failed: {doc_id}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _clear_progress(doc_id)


def _is_provider_busy(e: Exception) -> bool:
    return isinstance(e, PROVIDER_BUSY_ERRORS) or any(
        name in str(e) for name in ("RateLimitError", "ServiceUnavailableError")
    )


def _reset_rate_limit_strikes() -> None:
    global _rl_strikes
    with _rl_lock:
        _rl_strikes = 0


def build_media(doc_id: str) -> None:
    entry = store.load_registry().get(doc_id)
    if not entry or entry.get("type", "pdf") != "pdf":
        return
    store.update_document(doc_id, media_status="processing", media_error="")
    try:
        _check_api_keys(media.MODEL)
        figure_count, failed_pages = media.extract(
            doc_id, entry["pdf_path"],
            on_progress=lambda done, total: _set_progress(doc_id, "figures", done, total),
        )
        store.update_document(
            doc_id,
            media_status="done",
            media_error=f"Figure detection failed on page(s) {', '.join(map(str, failed_pages))}" if failed_pages else "",
            figure_count=figure_count,
            media_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        print(f"Media done: {doc_id} ({figure_count} figures)")
        _reset_rate_limit_strikes()
    except Exception as e:
        if _is_provider_busy(e):
            _rate_limit_cooldown("media", doc_id, e)
            return
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        store.update_document(doc_id, media_status="failed", media_error=msg)
        print(f"Media failed: {doc_id}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _clear_progress(doc_id)


def enqueue(doc_id: str) -> None:
    _queue.put(("tree", doc_id))


def enqueue_media(doc_id: str) -> None:
    store.update_document(doc_id, media_status="queued", media_error="")
    _queue.put(("media", doc_id))


def _worker() -> None:
    while True:
        kind, doc_id = _queue.get()
        try:
            _rate_limit_wait()
            if kind == "media":
                build_media(doc_id)
            else:
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
            _queue.put(("tree", doc_id))
            print(f"Re-queued after restart: {doc_id}")
        elif entry.get("media_status") in ("queued", "processing"):
            enqueue_media(doc_id)
            print(f"Re-queued media after restart: {doc_id}")


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    requeue_interrupted()
    for i in range(WORKERS):
        threading.Thread(target=_worker, daemon=True, name=f"ingest-worker-{i}").start()
