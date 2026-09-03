import logging
from functools import lru_cache

from mistralai.client import Mistral
from mistralai.client.errors import MistralError
from mistralai.client.utils import BackoffStrategy, RetryConfig

from app.config import settings

log = logging.getLogger(__name__)

# The SDK retries 429 and 5xx itself with this backoff (free/low tiers are rate-limited hard).
RETRY = RetryConfig(
    "backoff",
    BackoffStrategy(initial_interval=1000, max_interval=30_000, exponent=2.0, max_elapsed_time=120_000),
    retry_connection_errors=True,
)

# The embeddings endpoint caps the TOTAL tokens per request, so batches are
# packed by estimated size (~4 chars/token), not by a fixed count.
MAX_BATCH_CHARS = 40_000
MAX_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def get_client() -> Mistral:
    return Mistral(api_key=settings.mistral_api_key, retry_config=RETRY)


def _embed_batch(batch: list[str]) -> list[list[float]]:
    try:
        resp = get_client().embeddings.create(model=settings.embed_model, inputs=batch)
    except MistralError as exc:
        if "Too many tokens" in str(exc) and len(batch) > 1:
            log.warning("embedding batch of %d texts over the token cap, splitting it", len(batch))
            mid = len(batch) // 2
            return _embed_batch(batch[:mid]) + _embed_batch(batch[mid:])
        raise
    vectors = [d.embedding for d in resp.data]
    if any(v is None for v in vectors):
        raise RuntimeError("embeddings response is missing vectors")
    return [v for v in vectors if v is not None]


def _batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        if batch and (batch_chars + len(text) > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_SIZE):
            batches.append(batch)
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        batches.append(batch)
    return batches


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with mistral-embed (1024 dims), batched under the request token cap."""
    vectors: list[list[float]] = []
    for batch in _batches(texts):
        vectors.extend(_embed_batch(batch))
    return vectors
