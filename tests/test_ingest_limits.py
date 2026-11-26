"""Ingest and figure extraction must never run without end.

Each run sends paid LLM calls, so every way back into the queue - provider
busy, container restart, a runaway inside one run - has to stop at a hard
limit and leave the document "failed" until someone retries it on purpose.
"""
import asyncio
import os
import queue
import time
from types import SimpleNamespace

import litellm
import pytest
from pageindex import page_index_classic as tree_builder
from pageindex.utils import llm_completion

import documents
import jobs
import llm
import ocr
import store


@pytest.fixture
def isolated_queue(monkeypatch):
    """A queue no background worker reads, so tests see exactly what was put."""
    fresh = queue.Queue()
    monkeypatch.setattr(jobs, "_queue", fresh)
    monkeypatch.setattr(jobs, "_scheduled", set())
    return fresh


@pytest.fixture
def no_cooldown(monkeypatch):
    monkeypatch.setattr(jobs, "_rl_until", 0.0)
    monkeypatch.setattr(jobs, "_rl_strikes", 0)


@pytest.fixture
def pdf_doc(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    counter = iter(range(10_000))

    def make(**fields):
        doc_id = f"limits-{os.getpid()}-{next(counter)}"
        path = tmp_path / f"{doc_id}.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        store.create_document(doc_id, doc_name=f"{doc_id}.pdf", project="", pdf_path=str(path), page_count=4)
        if fields:
            store.update_document(doc_id, **fields)
        return doc_id

    return make


def rate_limited(*args, **kwargs):
    raise litellm.RateLimitError(message="quota exhausted", model="m", llm_provider="gemini")


def test_a_busy_provider_ends_in_failed_after_max_attempts(pdf_doc, monkeypatch, no_cooldown):
    monkeypatch.setattr(jobs, "get_page_tokens", rate_limited)
    doc_id = pdf_doc()

    for attempt in range(1, jobs.MAX_ATTEMPTS):
        assert jobs.build_tree(doc_id) is True
        entry = store.load_registry()[doc_id]
        assert entry["status"] == "queued"
        assert f"attempt {attempt} of {jobs.MAX_ATTEMPTS}" in entry["error"]

    assert jobs.build_tree(doc_id) is False
    entry = store.load_registry()[doc_id]
    assert entry["status"] == "failed"
    assert entry["attempts"] == jobs.MAX_ATTEMPTS
    assert "Gave up after" in entry["error"]

    calls = []
    monkeypatch.setattr(jobs, "get_page_tokens", lambda *a, **k: calls.append(1))
    assert jobs.build_tree(doc_id) is False
    assert calls == [], "a document out of attempts must not start another run"


def test_figure_extraction_is_bounded_the_same_way(pdf_doc, monkeypatch, no_cooldown):
    monkeypatch.setattr(jobs.media, "extract", rate_limited)
    monkeypatch.setattr(jobs.media, "MODEL", "gemini/test")
    doc_id = pdf_doc(status="done")

    results = [jobs.build_media(doc_id) for _ in range(jobs.MAX_ATTEMPTS + 2)]
    assert results == [True] * (jobs.MAX_ATTEMPTS - 1) + [False, False, False]
    entry = store.load_registry()[doc_id]
    assert entry["media_status"] == "failed"
    assert entry["media_attempts"] == jobs.MAX_ATTEMPTS


def test_a_run_that_kills_the_process_cannot_loop_through_restarts(pdf_doc, monkeypatch, isolated_queue):
    def crash(*args, **kwargs):
        raise KeyboardInterrupt  # stands in for an out-of-memory kill: nothing after it runs

    monkeypatch.setattr(jobs, "get_page_tokens", crash)
    doc_id = pdf_doc()

    restarts = 0
    while store.load_registry()[doc_id]["status"] != "failed":
        assert restarts <= jobs.MAX_ATTEMPTS, "restarts kept re-running the document"
        with pytest.raises(KeyboardInterrupt):
            jobs.build_tree(doc_id)
        assert store.load_registry()[doc_id]["status"] == "processing"
        jobs.requeue_interrupted()
        while not isolated_queue.empty():
            isolated_queue.get_nowait()
        jobs._scheduled.clear()
        restarts += 1

    entry = store.load_registry()[doc_id]
    assert restarts == jobs.MAX_ATTEMPTS
    assert "interrupted every time" in entry["error"]


def test_restart_requeues_only_what_still_has_attempts_left(pdf_doc, isolated_queue):
    resumable = pdf_doc(status="processing", attempts=1)
    exhausted = pdf_doc(status="processing", attempts=jobs.MAX_ATTEMPTS)
    failed = pdf_doc(status="failed", error="earlier failure", attempts=1)
    media_resumable = pdf_doc(status="done", media_status="queued", media_attempts=0)

    jobs.requeue_interrupted()
    jobs.requeue_interrupted()

    queued = []
    while not isolated_queue.empty():
        queued.append(isolated_queue.get_nowait())
    mine = {resumable, exhausted, failed, media_resumable}
    assert sorted(q for q in queued if q[1] in mine) == sorted([("tree", resumable), ("media", media_resumable)])

    registry = store.load_registry()
    assert registry[exhausted]["status"] == "failed"
    assert registry[failed]["status"] == "failed"
    assert registry[failed]["error"] == "earlier failure"


def test_a_document_is_never_queued_twice(isolated_queue, monkeypatch):
    assert jobs._put("tree", "same-doc") is True
    assert jobs._put("tree", "same-doc") is False
    assert jobs._put("media", "same-doc") is True
    assert isolated_queue.qsize() == 2

    monkeypatch.setattr(jobs, "build_tree", lambda doc_id: False)
    jobs._run(*isolated_queue.get_nowait())
    assert jobs._put("tree", "same-doc") is True, "a finished job must be schedulable again"


def test_the_worker_requeues_only_when_the_job_asks_for_it(isolated_queue, monkeypatch):
    monkeypatch.setattr(jobs, "build_tree", lambda doc_id: True)
    jobs._run("tree", "busy-doc")
    assert isolated_queue.get_nowait() == ("tree", "busy-doc")

    monkeypatch.setattr(jobs, "build_tree", lambda doc_id: False)
    jobs._run("tree", "busy-doc")
    assert isolated_queue.empty()


def test_an_explicit_retry_grants_a_fresh_budget(pdf_doc, isolated_queue):
    doc_id = pdf_doc(status="failed", error="Gave up after 3 attempt(s)", attempts=3, media_attempts=3)
    payload, status = documents.retry(doc_id)
    assert status == 200
    entry = store.load_registry()[doc_id]
    assert (entry["status"], entry["attempts"], entry["media_attempts"]) == ("queued", 0, 0)
    assert isolated_queue.get_nowait() == ("tree", doc_id)


def test_status_reports_the_attempts(pdf_doc):
    doc_id = pdf_doc(status="failed", attempts=2)
    payload, _ = documents.status(doc_id)
    assert (payload["attempts"], payload["max_attempts"]) == (2, jobs.MAX_ATTEMPTS)


def test_budget_stops_at_the_call_cap_and_stays_stopped():
    budget = llm.LLMBudget(max_calls=3, seconds=60)
    for _ in range(3):
        budget.charge()
    for _ in range(2):
        with pytest.raises(llm.LLMBudgetExceeded):
            budget.charge()


def test_budget_stops_at_the_time_limit():
    budget = llm.LLMBudget(max_calls=1000, seconds=0)
    time.sleep(0.01)
    with pytest.raises(llm.LLMBudgetExceeded):
        budget.charge()


def fake_response(text="transcribed page text that is long enough"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def test_the_budget_reaches_into_the_ocr_thread_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "MODEL", "gemini/test")
    monkeypatch.setattr(ocr, "_render_page_png", lambda pdf, i: b"png")
    monkeypatch.setattr(llm, "_completion", lambda **kw: calls.append(1) or fake_response())
    sparse_pages = [("", 0)] * 40

    token = llm.llm_budget.set(llm.LLMBudget(max_calls=5, seconds=60))
    try:
        with pytest.raises(llm.LLMBudgetExceeded):
            ocr.augment_page_list("unused.pdf", sparse_pages)
    finally:
        llm.llm_budget.reset(token)
    assert len(calls) == 5


def test_requests_that_cannot_succeed_are_not_retried(monkeypatch):
    calls = []

    def bad_request(**kwargs):
        calls.append(1)
        raise litellm.BadRequestError(message="prompt too long", model="m", llm_provider="gemini")

    monkeypatch.setattr(llm, "_completion", bad_request)
    with pytest.raises(litellm.BadRequestError):
        llm_completion("gemini/test", "prompt")
    assert len(calls) == 1


def test_every_request_carries_a_timeout(monkeypatch):
    sent = []

    async def async_fake(**kwargs):
        sent.append(kwargs)

    monkeypatch.setattr(llm, "_completion", lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(llm, "_acompletion", async_fake)
    litellm.completion(model="gemini/test", messages=[])
    asyncio.run(litellm.acompletion(model="gemini/test", messages=[]))
    assert len(sent) == 2 and all(kwargs["timeout"] > 0 for kwargs in sent)


def test_the_budget_covers_the_library_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_completion", lambda **kw: calls.append(1) or fake_response())

    token = llm.llm_budget.set(llm.LLMBudget(max_calls=2, seconds=60))
    try:
        llm_completion("gemini/test", "prompt")
        llm_completion("gemini/test", "prompt")
        with pytest.raises(Exception) as raised:
            llm_completion("gemini/test", "prompt")
    finally:
        llm.llm_budget.reset(token)
    assert len(calls) == 2
    assert isinstance(jobs._budget_exceeded(raised.value), llm.LLMBudgetExceeded)


def test_a_split_that_makes_no_progress_is_not_recursed_into(monkeypatch):
    calls = []

    async def same_range_split(node_page_list, mode=None, start_index=1, opt=None, logger=None):
        calls.append(start_index)
        if len(calls) > 3:
            raise AssertionError("recursed into the same page range again")
        return [{"structure": "1", "title": "Not the parent title", "physical_index": start_index,
                 "appear_start": "yes"}]

    async def passthrough(items, page_list, model=None, logger=None):
        return items

    monkeypatch.setattr(tree_builder, "meta_processor", same_range_split)
    monkeypatch.setattr(tree_builder, "check_title_appearance_in_start_concurrent", passthrough)
    opt = SimpleNamespace(max_page_num_each_node=10, max_token_num_each_node=20000, model=None)
    node = {"title": "Chapter", "start_index": 1, "end_index": 30}
    page_list = [("page text", 1000)] * 30

    asyncio.run(tree_builder.process_large_node_recursively(node, page_list, opt))
    assert calls == [1]
