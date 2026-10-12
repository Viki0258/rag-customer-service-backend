from __future__ import annotations

import hashlib

from whatsapp_support.domain import Document, KnowledgeChunk
from whatsapp_support.ports import EmbeddingPort, KnowledgeStorePort


def chunk_text(text: str, max_chars: int) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("document text must be a string")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    paragraphs = [
        paragraph.strip()
        for paragraph in text.replace("\r\n", "\n").split("\n\n")
    ]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        remaining = paragraph
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
    return chunks


def make_chunk(document: Document, ordinal: int, text: str) -> KnowledgeChunk:
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    stable_material = f"{document.document_id}:{document.checksum}:{ordinal}:{checksum}"
    chunk_id = (
        f"{document.document_id}:{ordinal}:"
        f"{hashlib.sha256(stable_material.encode('utf-8')).hexdigest()[:16]}"
    )
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        ordinal=ordinal,
        text=text,
        embedding_input=text,
        checksum=checksum,
        metadata=dict(document.metadata),
    )


class KnowledgeIndexer:
    def __init__(
        self,
        embedding: EmbeddingPort,
        store: KnowledgeStorePort,
        *,
        max_chars: int = 1200,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._embedding = embedding
        self._store = store
        self._max_chars = max_chars

    async def index(self, document: Document) -> list[KnowledgeChunk]:
        texts = chunk_text(document.text, self._max_chars)
        if not texts:
            return []

        chunks = [make_chunk(document, ordinal, text) for ordinal, text in enumerate(texts)]
        embeddings = await self._embedding.embed(
            [chunk.embedding_input for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise ValueError("embedding count does not match chunk count")
        await self._store.upsert_chunks(chunks)
        return chunks
