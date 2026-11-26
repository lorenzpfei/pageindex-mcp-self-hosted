import contextvars
import os
import threading
import time

import litellm


def litellm_model(model):
    if model and model.startswith("google/"):
        return "gemini/" + model.removeprefix("google/")
    return model


def completion_kwargs(model):
    """Gemini 3 models degrade with temperature < 1.0 (LiteLLM warns about
    infinite loops / failures); keep the deterministic 0 only elsewhere."""
    if model and model.removeprefix("litellm/").startswith("gemini"):
        return {}
    return {"temperature": 0}


def retry_sleep(attempt):
    """Exponential backoff: rate-limit errors (e.g. Gemini free tier) come
    with retry windows of 30s+ that a 1s sleep never outlasts."""
    return min(2 ** attempt, 60)


# Cap concurrent LLM calls, shared across the OCR and figure threads.
MAX_CONCURRENT_LLM = max(1, int(os.getenv("PAGEINDEX_MAX_CONCURRENT_LLM", "8")))
_llm_slots = threading.BoundedSemaphore(MAX_CONCURRENT_LLM)

# Errors meaning "the provider can't serve anyone right now": 429 (quota /
# rate limit) and 503 (model overloaded). Per-page or per-section degradation
# makes no sense for these - every remaining call would fail too - so callers
# abort the ingest and let jobs.py re-queue the document with a cooldown.
PROVIDER_BUSY_ERRORS = (litellm.RateLimitError, litellm.ServiceUnavailableError)

# Errors no retry can fix: bad key, rejected or oversized request, unknown
# model. Retrying them only resends the full prompt for nothing.
NON_RETRYABLE_ERRORS = (
    litellm.AuthenticationError,
    litellm.BadRequestError,
    litellm.NotFoundError,
    litellm.PermissionDeniedError,
)

# Without it a stalled request holds a worker for LiteLLM's default of well
# over an hour, and the per-job time limit below is only checked between calls.
LLM_CALL_TIMEOUT = 600


class LLMBudgetExceeded(Exception):
    """The running job used up its LLM call budget or its time limit."""


class LLMBudget:
    """Hard cap on the LLM calls of one ingest or figure-extraction attempt.

    Every request counts, retries included, because every request can be
    billed. Once exhausted it stays exhausted, so parallel calls of the same
    job stop at their next request instead of finishing their retries.
    """

    def __init__(self, max_calls: int, seconds: float):
        self.max_calls = max_calls
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds
        self.calls = 0
        self._lock = threading.Lock()

    def charge(self) -> None:
        with self._lock:
            if time.monotonic() > self.deadline:
                raise LLMBudgetExceeded(f"stopped after the time limit of {round(self.seconds / 60)} min")
            if self.calls >= self.max_calls:
                raise LLMBudgetExceeded(f"stopped after the budget of {self.max_calls} LLM calls")
            self.calls += 1


# Set by jobs.py around each attempt. Thread pools must carry it over with
# contextvars.copy_context(); asyncio tasks inherit it on their own.
llm_budget = contextvars.ContextVar("pageindex_llm_budget", default=None)


def charge_llm_call() -> None:
    budget = llm_budget.get()
    if budget is not None:
        budget.charge()


_completion = litellm.completion
_acompletion = litellm.acompletion


def _charged_completion(*args, **kwargs):
    charge_llm_call()
    return _completion(*args, **{"timeout": LLM_CALL_TIMEOUT, **kwargs})


async def _charged_acompletion(*args, **kwargs):
    charge_llm_call()
    return await _acompletion(*args, **{"timeout": LLM_CALL_TIMEOUT, **kwargs})


# pageindex looks these up on the module at call time, so its indexing calls are charged here too.
litellm.completion = _charged_completion
litellm.acompletion = _charged_acompletion


def strip_surrogates(text):
    # PyPDF2 decodes some glyphs to lone surrogates, which cannot be UTF-8 encoded into LLM requests.
    return (text or "").encode("utf-8", "ignore").decode("utf-8")
