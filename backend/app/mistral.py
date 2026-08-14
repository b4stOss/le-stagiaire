import time
from functools import lru_cache

from mistralai.client import Mistral

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> Mistral:
    return Mistral(api_key=settings.mistral_api_key)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts with mistral-embed (1024 dims), batched, with basic backoff on 429s."""
    client = get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=settings.embed_model, inputs=batch)
                vectors.extend(d.embedding for d in resp.data)
                break
            except Exception as exc:  # SDK raises typed errors; retry on rate limits, fail fast otherwise
                if "429" in str(exc) and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise
    return vectors
