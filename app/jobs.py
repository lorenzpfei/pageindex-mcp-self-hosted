"""Background ingest workers.

A small pool of daemon threads (PAGEINDEX_INGEST_WORKERS, default 2) builds
trees for uploaded PDFs. Each ingest holds the PDF + tree in memory
(~150-200 MB for typical lecture decks) and page_index_main already
parallelizes its LLM calls internally, so a high worker count mostly burns
RAM and OpenAI rate limits rather than speeding things up.

Every run of a job is an attempt that can cost real money, so each one is
bounded three ways: at most MAX_ATTEMPTS per document and job kind (counted
in the registry, so container restarts count too), an LLM call budget and a
time limit per attempt. A document that exhausts its attempts ends up
"failed" for good - only an explicit retry gives it a fresh budget.
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from pageindex.flash import page_index_flash
from pageindex.flash.api import flash_rejection_reason
from pageindex.page_index_classic import page_index_main
from pageindex.utils import (
    ConfigLoader,
    LLMRetriesExhausted,
    _llm_usage,
    create_clean_structure_for_description,
    format_structure,
    generate_doc_description,
    get_page_tokens,
    write_node_id,
)

import media
import ocr
import search
import store
from llm import (
    MAX_CONCURRENT_LLM, PROVIDER_BUSY_ERRORS, LLMBudget, LLMBudgetExceeded, litellm_model, llm_budget,
    strip_surrogates,
)

WORKERS = max(1, int(os.environ.get("PAGEINDEX_INGEST_WORKERS", "2")))
MODEL = litellm_model(os.environ.get("PAGEINDEX_MODEL") or "gemini/gemini-4-hydrogen")
SUMMARY_MODEL = litellm_model(os.environ.get("PAGEINDEX_SUMMARY_MODEL") or MODEL)
INDEX_MODE = os.environ.get("PAGEINDEX_INDEX_MODE") or "auto"
if INDEX_MODE not in ("auto", "standard"):
    raise ValueError(f"PAGEINDEX_INDEX_MODE must be 'auto' or 'standard', got {INDEX_MODE!r}")
MAX_OCR_SHARE_FOR_FLASH = 0.1

MAX_ATTEMPTS = 3
# Per attempt. A normal ingest needs a handful of calls per page; this leaves
# room for retries but stops a runaway long before it gets expensive.
LLM_CALLS_PER_PAGE = 25
LLM_CALLS_MIN = 200
ATTEMPT_TIMEOUT = 3 * 3600

# registry fields per job kind: (status, error, attempt counter)
_FIELDS = {
    "tree": ("status", "error", "attempts"),
    "media": ("media_status", "media_error", "media_attempts"),
}

_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_started = False

# Jobs waiting or running. A job is only put on the queue when it is in
# neither state, so a document never runs twice at once.
_scheduled_lock = threading.Lock()
_scheduled: set[tuple[str, str]] = set()

# Flash parses with one spawned process per CPU; two at once would not fit the container's memory limit.
_flash_lock = threading.Lock()

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


def _provider_busy(kind: str, doc_id: str, e: Exception) -> bool:
    """Start a cooldown and tell the worker whether to re-queue the document."""
    global _rl_until, _rl_strikes
    with _rl_lock:
        _rl_strikes += 1
        cooldown = min(60 * 2 ** _rl_strikes, RL_COOLDOWN_MAX)
        _rl_until = max(_rl_until, time.time() + cooldown)
    status_key, error_key, attempts_key = _FIELDS[kind]
    attempts = (store.load_registry().get(doc_id) or {}).get(attempts_key) or 0
    if attempts >= MAX_ATTEMPTS:
        _give_up(kind, doc_id, attempts, f"LLM provider still busy (rate limit, quota or overload) - retry later ({e})")
        return False
    message = (
        f"LLM provider busy (rate limit/overload) - retrying automatically in ~{cooldown // 60} min, "
        f"attempt {attempts} of {MAX_ATTEMPTS} ({e})"
    )
    store.update_document(doc_id, **{status_key: "queued", error_key: message[:500]})
    print(f"Rate limited during {doc_id}; re-queued, pausing ingest {cooldown}s", file=sys.stderr)
    return True


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


def _give_up(kind: str, doc_id: str, attempts: int, reason: str) -> None:
    status_key, error_key, _ = _FIELDS[kind]
    message = f"Gave up after {attempts} attempt(s): {reason} - use retry to try again"
    store.update_document(doc_id, **{status_key: "failed", error_key: message[:500]})
    print(f"Gave up on {kind} for {doc_id} after {attempts} attempt(s)", file=sys.stderr)


INTERRUPTED = "interrupted every time (container restart or out-of-memory kill?)"


def _begin_attempt(kind: str, doc_id: str, entry: dict) -> bool:
    """Count one more attempt before any LLM call. Counting up front is what
    bounds crash loops: a run that kills the container still used up its
    attempt. False (document marked failed) once all attempts are spent."""
    status_key, error_key, attempts_key = _FIELDS[kind]
    attempts = entry.get(attempts_key) or 0
    if attempts >= MAX_ATTEMPTS:
        _give_up(kind, doc_id, attempts, entry.get(error_key) or INTERRUPTED)
        return False
    store.update_document(doc_id, **{status_key: "processing", error_key: "", attempts_key: attempts + 1})
    return True


def _attempt_budget(entry: dict) -> LLMBudget:
    pages = entry.get("page_count") or 0
    return LLMBudget(max(LLM_CALLS_MIN, LLM_CALLS_PER_PAGE * pages), ATTEMPT_TIMEOUT)


def build_tree(doc_id: str) -> bool:
    """Run PageIndex tree building for a registered document (blocking).
    Returns True when the document should be queued again."""
    entry = store.load_registry().get(doc_id)
    if not entry or not _begin_attempt("tree", doc_id, entry):
        return False
    budget = llm_budget.set(_attempt_budget(entry))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
             "completion_tokens_details": {"reasoning_tokens": 0}}
    usage_token = _llm_usage.set(usage)
    try:
        _check_api_keys(SUMMARY_MODEL, *([MODEL] if INDEX_MODE == "standard" else []),
                        *([ocr.MODEL] if ocr.enabled() else []))

        _set_progress(doc_id, "reading")
        page_list = [(strip_surrogates(text), tokens) for text, tokens in get_page_tokens(entry["pdf_path"], model=MODEL)]
        page_list, ocr_pages = ocr.augment_page_list(
            entry["pdf_path"], page_list, model=MODEL,
            on_progress=lambda done, total: _set_progress(doc_id, "ocr", done, total),
        )
        _set_progress(doc_id, "tree")
        if ocr_pages:
            print(f"OCR transcribed {ocr_pages} text-poor page(s) for {doc_id}")

        result = None
        if INDEX_MODE == "auto" and ocr_pages <= MAX_OCR_SHARE_FOR_FLASH * len(page_list):
            result = _index_flash(entry["pdf_path"], len(page_list))
        index_mode = "flash" if result else "standard"
        if not result:
            result = _index_standard(entry["pdf_path"], page_list)
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
            index_mode=index_mode,
            usage={key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        )
        print(f"Ingest done: {doc_id} ({index_mode}, {usage['total_tokens']} tokens)")
        try:
            search.build(doc_id)
        except Exception as e:
            print(f"Search cache failed for {doc_id}: {e}")
        _reset_rate_limit_strikes()
        if media.enabled():
            enqueue_media(doc_id)
    except Exception as e:
        e = _budget_exceeded(e) or e
        if _is_provider_busy(e):
            return _provider_busy("tree", doc_id, e)
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        store.update_document(doc_id, status="failed", error=msg[:500])
        print(f"Ingest failed: {doc_id}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _llm_usage.reset(usage_token)
        llm_budget.reset(budget)
        _clear_progress(doc_id)
    return False


def _index_flash(pdf_path: str, page_count: int) -> dict | None:
    with _flash_lock:
        result = page_index_flash(
            pdf_path, summary=True, summary_model=SUMMARY_MODEL, optimize="full",
            optimize_model=SUMMARY_MODEL, summary_concurrency=MAX_CONCURRENT_LLM,
        )
    structure = result.get("structure", [])
    reason = flash_rejection_reason(result)
    if reason or not _within_pages(structure, page_count):
        print(f"Flash index rejected for {pdf_path}, falling back to standard: {reason or 'pages out of range'}")
        return None
    write_node_id(structure)
    description = generate_doc_description(
        create_clean_structure_for_description(structure, top_level_only=True), model=SUMMARY_MODEL,
    )
    return {
        "doc_name": result.get("doc_name", ""),
        "doc_description": description,
        "structure": format_structure(
            structure, order=["title", "node_id", "start_index", "end_index", "key_items", "summary", "nodes"],
        ),
    }


def _within_pages(nodes: list, page_count: int) -> bool:
    for node in nodes or []:
        start, end = node.get("start_index"), node.get("end_index")
        if start is not None and end is not None and not (1 <= start and end <= page_count):
            return False
        if not _within_pages(node.get("nodes"), page_count):
            return False
    return True


def _index_standard(pdf_path: str, page_list: list) -> dict:
    _check_api_keys(MODEL)
    opt = ConfigLoader().load({"model": MODEL, "summary_model": SUMMARY_MODEL, "if_add_doc_description": "yes"})
    return page_index_main(pdf_path, opt, page_list=page_list)


def _is_provider_busy(e: Exception) -> bool:
    if isinstance(e, LLMRetriesExhausted) and e.status_code in (429, 503):
        return True
    return isinstance(e, PROVIDER_BUSY_ERRORS) or any(
        name in str(e) for name in ("RateLimitError", "ServiceUnavailableError")
    )


def _budget_exceeded(e: BaseException | None) -> LLMBudgetExceeded | None:
    while e is not None and not isinstance(e, LLMBudgetExceeded):
        e = e.__cause__ or e.__context__
    return e


def _reset_rate_limit_strikes() -> None:
    global _rl_strikes
    with _rl_lock:
        _rl_strikes = 0


def build_media(doc_id: str) -> bool:
    """Extract the figures of a PDF (blocking). Returns True when the
    document should be queued again."""
    entry = store.load_registry().get(doc_id)
    if not entry or entry.get("type", "pdf") != "pdf" or not _begin_attempt("media", doc_id, entry):
        return False
    budget = llm_budget.set(_attempt_budget(entry))
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
            return _provider_busy("media", doc_id, e)
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        store.update_document(doc_id, media_status="failed", media_error=msg[:500])
        print(f"Media failed: {doc_id}", file=sys.stderr)
        traceback.print_exc()
    finally:
        llm_budget.reset(budget)
        _clear_progress(doc_id)
    return False


def _put(kind: str, doc_id: str) -> bool:
    with _scheduled_lock:
        if (kind, doc_id) in _scheduled:
            return False
        _scheduled.add((kind, doc_id))
    _queue.put((kind, doc_id))
    return True


def enqueue(doc_id: str) -> bool:
    return _put("tree", doc_id)


def enqueue_media(doc_id: str) -> bool:
    store.update_document(doc_id, media_status="queued", media_error="")
    return _put("media", doc_id)


def _run(kind: str, doc_id: str) -> None:
    try:
        _rate_limit_wait()
        requeue = build_media(doc_id) if kind == "media" else build_tree(doc_id)
    finally:
        with _scheduled_lock:
            _scheduled.discard((kind, doc_id))
    if requeue:
        _put(kind, doc_id)


def _worker() -> None:
    while True:
        kind, doc_id = _queue.get()
        try:
            _run(kind, doc_id)
        except Exception:
            traceback.print_exc()
        finally:
            _queue.task_done()


def requeue_interrupted() -> None:
    """Re-enqueue documents left in "queued"/"processing" by a restart.

    The queue only lives in memory, so a crash (e.g. an OOM kill) would
    otherwise silently drop everything that was still waiting. Documents
    that already used up their attempts are marked failed instead - they
    are most likely the reason the container keeps restarting. "failed"
    documents are never picked up here.
    """
    for doc_id, entry in store.load_registry().items():
        for kind, (status_key, error_key, attempts_key) in _FIELDS.items():
            if entry.get(status_key) not in ("queued", "processing"):
                continue
            if kind == "media" and entry.get("type", "pdf") != "pdf":
                continue
            attempts = entry.get(attempts_key) or 0
            if attempts >= MAX_ATTEMPTS:
                _give_up(kind, doc_id, attempts, entry.get(error_key) or INTERRUPTED)
            elif _put(kind, doc_id):
                store.update_document(doc_id, **{status_key: "queued"})
                print(f"Re-queued {kind} after restart: {doc_id} (attempt {attempts} of {MAX_ATTEMPTS} used)")
            break


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    requeue_interrupted()
    for i in range(WORKERS):
        threading.Thread(target=_worker, daemon=True, name=f"ingest-worker-{i}").start()
