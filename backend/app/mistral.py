import time
from functools import lru_cache

from mistralai.client import Mistral

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> Mistral:
    return Mistral(api_key=settings.mistral_api_key)


# The embeddings endpoint caps the TOTAL tokens per request, so batches are
# packed by estimated size (~4 chars/token), not by a fixed count.
MAX_BATCH_CHARS = 40_000
MAX_BATCH_SIZE = 64


def _embed_batch(batch: list[str]) -> list[list[float]]:
    client = get_client()
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=settings.embed_model, inputs=batch)
            return [d.embedding for d in resp.data]
        except Exception as exc:
            msg = str(exc)
            if "Too many tokens" in msg and len(batch) > 1:
                mid = len(batch) // 2  # our estimate was off: halve and retry
                return _embed_batch(batch[:mid]) + _embed_batch(batch[mid:])
            if "429" in msg and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with mistral-embed (1024 dims), batched under the request token cap."""
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        if batch and (batch_chars + len(text) > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_SIZE):
            vectors.extend(_embed_batch(batch))
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        vectors.extend(_embed_batch(batch))
    return vectors
