from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import Document
from whatsapp_support.fakes import FakeEmbedding, FakeKnowledgeStore
from whatsapp_support.indexer import KnowledgeIndexer


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


def make_document(text: str) -> Document:
    return Document(
        document_id="returns-v1",
        source_uri="fixture://returns-v1",
        title="Returns policy",
        text=text,
        content_type="text/plain",
        checksum="document-checksum-v1",
        created_at=NOW,
        metadata={"locale": "en", "market": "US", "version": "v1"},
    )


def test_indexer_chunks_paragraphs_preserves_metadata_and_is_stable() -> None:
    async def scenario() -> None:
        embedding = FakeEmbedding()
        store = FakeKnowledgeStore()
        indexer = KnowledgeIndexer(embedding, store)
        document = make_document("First paragraph.\n\nSecond paragraph.")

        chunks = await indexer.index(document)
        again = await indexer.index(document)

        assert [chunk.text for chunk in chunks] == [
            "First paragraph.",
            "Second paragraph.",
        ]
        assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in again]
        assert all(chunk.metadata == document.metadata for chunk in chunks)
        assert embedding.calls[0] == ["First paragraph.", "Second paragraph."]
        assert store.calls[0] == chunks

    asyncio.run(scenario())


def test_indexer_splits_long_paragraphs_and_rejects_embedding_count_mismatch() -> None:
    async def scenario() -> None:
        embedding = FakeEmbedding()
        store = FakeKnowledgeStore()
        indexer = KnowledgeIndexer(embedding, store, max_chars=10)
        chunks = await indexer.index(make_document("abcdefghijklmno"))
        assert chunks
        assert all(len(chunk.text) <= 10 for chunk in chunks)

        broken_embedding = FakeEmbedding()
        broken_embedding.results_queue.append([])
        broken_store = FakeKnowledgeStore()
        broken_indexer = KnowledgeIndexer(broken_embedding, broken_store)
        with pytest.raises(ValueError, match="embedding count"):
            await broken_indexer.index(make_document("one paragraph"))
        assert broken_store.calls == []

    asyncio.run(scenario())
