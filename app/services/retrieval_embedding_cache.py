from __future__ import annotations

from hashlib import sha256
from typing import Sequence

from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import RetrievalEmbeddingCacheEntry
from app.services.memory_similarity import EmbeddingBatch, MemoryEmbeddingProvider


class CachedRetrievalEmbeddingProvider:
    """Persist embeddings without storing another copy of the source text."""

    def __init__(self, bind: Engine, delegate: MemoryEmbeddingProvider) -> None:
        self._bind = bind
        self._delegate = delegate
        self.provider_version = delegate.provider_version

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch([], self.provider_version, _zero_usage())

        hashes = [_content_hash(text) for text in texts]
        unique_text_by_hash = dict(zip(hashes, texts, strict=True))
        with Session(self._bind) as cache_session:
            cached = {
                entry.content_sha256: list(entry.vector)
                for entry in cache_session.scalars(
                    select(RetrievalEmbeddingCacheEntry).where(
                        RetrievalEmbeddingCacheEntry.provider_version
                        == self.provider_version,
                        RetrievalEmbeddingCacheEntry.content_sha256.in_(
                            unique_text_by_hash
                        ),
                    )
                )
            }

            missing_hashes = [
                content_hash
                for content_hash in unique_text_by_hash
                if content_hash not in cached
            ]
            usage = _zero_usage()
            usage["cache_hits"] = sum(content_hash in cached for content_hash in hashes)
            usage["cache_misses"] = len(missing_hashes)
            if missing_hashes:
                batch = self._delegate.embed(
                    [unique_text_by_hash[content_hash] for content_hash in missing_hashes]
                )
                if batch.provider_version != self.provider_version:
                    raise ValueError("embedding provider version changed during the request")
                if len(batch.vectors) != len(missing_hashes):
                    raise ValueError("embedding provider returned the wrong number of vectors")
                usage.update(batch.usage)
                new_entries = []
                for content_hash, vector in zip(
                    missing_hashes, batch.vectors, strict=True
                ):
                    if not vector:
                        raise ValueError("embedding vectors must have non-zero dimensions")
                    cached[content_hash] = list(vector)
                    new_entries.append(
                        RetrievalEmbeddingCacheEntry(
                            provider_version=self.provider_version,
                            content_sha256=content_hash,
                            vector=list(vector),
                            dimensions=len(vector),
                        )
                    )
                cache_session.add_all(new_entries)
                try:
                    cache_session.commit()
                except IntegrityError:
                    # Another worker may have inserted the same immutable cache keys.
                    cache_session.rollback()

        return EmbeddingBatch(
            vectors=[cached[content_hash] for content_hash in hashes],
            provider_version=self.provider_version,
            usage=usage,
        )


def _content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _zero_usage() -> dict[str, int]:
    return {
        "embedding_calls": 0,
        "embedding_tokens": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
