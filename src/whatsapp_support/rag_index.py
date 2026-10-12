from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import aiosqlite

from whatsapp_support.domain import Document, RetrievalQuery, RetrievedChunk
from whatsapp_support.indexer import chunk_text, make_chunk
from whatsapp_support.rag_retrieval import HybridRetriever, normalize_locale
from whatsapp_support.vector_validation import validate_vector


_SUPPORTED_SUFFIXES = {".md", ".txt"}
_LOCALE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Z0-9]{2,8})*$")
_DEFAULT_INDEX_NAME = "knowledge.sqlite3"


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    embedding_model: str
    embedding_dimension: int
    created_at: datetime


@dataclass(frozen=True)
class RagSearchConfig:
    locale: str
    candidate_k: int = 20
    final_k: int = 5
    threshold: float = 0.0
    vector_candidate_k: int | None = None
    bm25_candidate_k: int | None = None
    allowed_document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if self.final_k < 1:
            raise ValueError("final_k must be positive")
        if self.threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if self.vector_candidate_k is not None and self.vector_candidate_k < 1:
            raise ValueError("vector_candidate_k must be positive")
        if self.bm25_candidate_k is not None and self.bm25_candidate_k < 1:
            raise ValueError("bm25_candidate_k must be positive")

    @property
    def effective_vector_candidate_k(self) -> int:
        return self.vector_candidate_k or self.candidate_k

    @property
    def effective_bm25_candidate_k(self) -> int:
        return self.bm25_candidate_k or self.candidate_k


class KnowledgeFileLoader:
    @staticmethod
    def load_directory(path: str | Path) -> list[Document]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(str(root))
        if not root.is_dir():
            raise NotADirectoryError(str(root))

        documents: list[Document] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            documents.append(_load_document(file_path, root))
        return documents


class SqliteKnowledgeIndex:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._read_only = False
        self._live_readonly = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        if self._connection is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._connection.execute("PRAGMA journal_mode = WAL")
        await self._connection.executescript(_SCHEMA_SQL)

    async def open_readonly(self, *, live_updates: bool = False) -> None:
        if self._connection is not None:
            return
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        self._read_only = True
        self._live_readonly = live_updates
        connection: aiosqlite.Connection | None = None
        try:
            connection = await aiosqlite.connect(
                self._readonly_uri(), uri=True, isolation_level=None
            )
            connection.row_factory = sqlite3.Row
            await connection.execute("PRAGMA query_only = ON")
            self._connection = connection
        except Exception:
            if connection is not None:
                await connection.close()
            self._read_only = False
            self._connection = None
            raise

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None
        self._read_only = False

    async def active_generation(self) -> GenerationRecord | None:
        if self._connection is None:
            raise RuntimeError("index is not open")
        conn = await self._read_connection()
        try:
            await conn.execute("BEGIN")
            cursor = await conn.execute(
                """
                SELECT generation_id, embedding_model, embedding_dimension, created_at
                FROM generations
                WHERE active = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = await cursor.fetchone()
            await conn.commit()
        finally:
            await conn.close()
        if row is None:
            return None
        return GenerationRecord(
            generation_id=row["generation_id"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def import_directory(
        self,
        path: str | Path,
        embedding,
        locale: str | None = None,
    ) -> GenerationRecord:
        if self._connection is None:
            raise RuntimeError("index is not open")

        model_name, embedding_dimension = _embedding_metadata(embedding)
        documents = KnowledgeFileLoader.load_directory(path)
        if locale is not None:
            documents = [_override_locale(doc, locale) for doc in documents]
        chunk_records = await self._build_chunk_records(
            documents,
            embedding,
            model_name=model_name,
            dimension=embedding_dimension,
        )

        async with self._lock:
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                generation_id = _generation_id()
                created_at = datetime.now(UTC)
                await self._connection.execute(
                    """
                    INSERT INTO generations (
                        generation_id, embedding_model, embedding_dimension, created_at, active
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (
                        generation_id,
                        model_name,
                        embedding_dimension,
                        created_at.isoformat(),
                    ),
                )

                for document in documents:
                    await self._connection.execute(
                        """
                        INSERT INTO documents (
                            generation_id, document_id, source_uri, title,
                            locale, checksum, content_type, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            generation_id,
                            document.document_id,
                            document.source_uri,
                            document.title,
                            document.metadata["locale"],
                            document.checksum,
                            document.content_type,
                            json.dumps(
                                document.metadata,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        ),
                    )
                for chunk in chunk_records:
                    vector = validate_vector(
                        chunk["vector"],
                        embedding_dimension,
                        label="embedding vector",
                    )
                    if chunk["cache_write"]:
                        await self._connection.execute(
                            """
                            INSERT OR REPLACE INTO embedding_cache (
                                checksum, model_name, embedding_dimension, vector_json
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                chunk["checksum"],
                                model_name,
                                embedding_dimension,
                                json.dumps(vector, ensure_ascii=False),
                            ),
                        )
                    await self._connection.execute(
                        """
                        INSERT INTO chunks (
                            generation_id, chunk_id, document_id, locale, text,
                            checksum, metadata_json, vector_json, ordinal
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            generation_id,
                            chunk["chunk_id"],
                            chunk["document_id"],
                            chunk["locale"],
                            chunk["text"],
                            chunk["checksum"],
                            json.dumps(chunk["metadata"], ensure_ascii=False, sort_keys=True),
                            json.dumps(vector, ensure_ascii=False),
                            chunk["ordinal"],
                        ),
                    )
                await self._connection.execute("UPDATE generations SET active = 0 WHERE active = 1")
                await self._connection.execute(
                    "UPDATE generations SET active = 1 WHERE generation_id = ?",
                    (generation_id,),
                )
                await self._connection.commit()
            except BaseException:
                try:
                    await asyncio.shield(self._connection.rollback())
                except BaseException:
                    await asyncio.shield(self.close())
                raise

        await self.prune_old_generations(keep=2)
        return GenerationRecord(
            generation_id=generation_id,
            embedding_model=model_name,
            embedding_dimension=embedding_dimension,
            created_at=created_at,
        )

    async def search(
        self,
        query: str,
        embedding,
        config: RagSearchConfig,
    ) -> list[RetrievedChunk]:
        if self._connection is None:
            raise RuntimeError("index is not open")
        model_name, dimension = _embedding_metadata(embedding)
        query_vectors = await embedding.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("embedding count does not match query count")
        query_vector = validate_vector(
            query_vectors[0], dimension, label="query vector"
        )
        conn = await self._read_connection()
        try:
            await conn.execute("BEGIN")
            generation_cursor = await conn.execute(
                """
                SELECT generation_id, embedding_model, embedding_dimension
                FROM generations
                WHERE active = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            generation_row = await generation_cursor.fetchone()
            if generation_row is None:
                await conn.commit()
                return []
            if (
                generation_row["embedding_model"] != model_name
                or generation_row["embedding_dimension"] != dimension
            ):
                raise ValueError("active generation embedding mismatch")
            cursor = await conn.execute(
                """
                SELECT chunk_id, document_id, locale, text, metadata_json,
                       vector_json, ordinal
                FROM chunks
                WHERE generation_id = ?
                ORDER BY ordinal, chunk_id
                """,
                (generation_row["generation_id"],),
            )
            rows = await cursor.fetchall()
            await conn.commit()
        finally:
            await conn.close()
        return _retrieve_chunks(
            rows=rows,
            query=query,
            query_vector=query_vector,
            config=config,
        )

    async def search_vector(
        self,
        *,
        query: str,
        query_vector: Sequence[float],
        embedding_model: str,
        embedding_dimension: int,
        config: RagSearchConfig,
    ) -> list[RetrievedChunk]:
        if self._connection is None:
            raise RuntimeError("index is not open")
        query_vector = validate_vector(
            query_vector, embedding_dimension, label="query vector"
        )
        conn = await self._read_connection()
        try:
            await conn.execute("BEGIN")
            generation_cursor = await conn.execute(
                """
                SELECT generation_id, embedding_model, embedding_dimension
                FROM generations
                WHERE active = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            generation_row = await generation_cursor.fetchone()
            if generation_row is None:
                await conn.commit()
                return []
            if (
                generation_row["embedding_model"] != embedding_model
                or generation_row["embedding_dimension"] != embedding_dimension
            ):
                raise ValueError("active generation embedding mismatch")
            generation_id = generation_row["generation_id"]
            cursor = await conn.execute(
                """
                SELECT chunk_id, document_id, locale, text, metadata_json,
                       vector_json, ordinal
                FROM chunks
                WHERE generation_id = ?
                ORDER BY ordinal, chunk_id
                """,
                (generation_id,),
            )
            rows = await cursor.fetchall()
            await conn.commit()
        finally:
            await conn.close()

        return _retrieve_chunks(
            rows=rows,
            query=query,
            query_vector=query_vector,
            config=config,
        )

    async def inspect_active(self) -> dict:
        """Bounded read-only snapshot for local diagnostics; never export vectors/paths."""
        if self._connection is None:
            raise RuntimeError("index is not open")
        conn = await self._read_connection()
        try:
            await conn.execute("BEGIN")
            cursor = await conn.execute(
                "SELECT generation_id, embedding_model, embedding_dimension, created_at "
                "FROM generations WHERE active = 1 ORDER BY created_at DESC LIMIT 1")
            generation = await cursor.fetchone()
            if generation is None:
                return {"generation": None, "documents": [], "chunks": []}
            generation_id = generation['generation_id']
            cursor = await conn.execute(
                "SELECT document_id, title, locale, checksum, content_type FROM documents "
                "WHERE generation_id = ? ORDER BY title, document_id LIMIT 1001", (generation_id,))
            documents = await cursor.fetchall()
            cursor = await conn.execute(
                "SELECT chunk_id, document_id, locale, text, ordinal FROM chunks "
                "WHERE generation_id = ? ORDER BY document_id, ordinal, chunk_id LIMIT 5001", (generation_id,))
            chunks = await cursor.fetchall()
            if len(documents) > 1000 or len(chunks) > 5000:
                raise ValueError("Diagnostic snapshot exceeds demo limits")
            return {"generation": dict(generation),
                    "documents": [dict(row) for row in documents],
                    "chunks": [dict(row) for row in chunks]}
        finally:
            await conn.close()

    async def prune_old_generations(self, keep: int = 2) -> int:
        if self._connection is None:
            raise RuntimeError("index is not open")
        keep = max(2, keep)
        async with self._lock:
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._connection.execute(
                    """
                    SELECT generation_id
                    FROM generations
                    ORDER BY created_at DESC
                    """
                )
                generation_ids = [row[0] for row in await cursor.fetchall()]
                if len(generation_ids) <= keep:
                    await self._connection.rollback()
                    return 0
                removable = generation_ids[keep:]
                if not removable:
                    await self._connection.rollback()
                    return 0
                await self._connection.executemany(
                    "DELETE FROM chunks WHERE generation_id = ?",
                    [(generation_id,) for generation_id in removable],
                )
                await self._connection.executemany(
                    "DELETE FROM documents WHERE generation_id = ?",
                    [(generation_id,) for generation_id in removable],
                )
                await self._connection.executemany(
                    "DELETE FROM generations WHERE generation_id = ?",
                    [(generation_id,) for generation_id in removable],
                )
                await self._connection.commit()
                return len(removable)
            except BaseException:
                try:
                    await asyncio.shield(self._connection.rollback())
                except BaseException:
                    await asyncio.shield(self.close())
                raise

    async def _build_chunk_records(
        self,
        documents: Sequence[Document],
        embedding,
        *,
        model_name: str,
        dimension: int,
    ) -> list[dict[str, object]]:
        chunk_entries: list[dict[str, object]] = []
        for document in documents:
            for ordinal, text in enumerate(chunk_text(document.text, 1200)):
                chunk = make_chunk(document, ordinal, text)
                locale = document.metadata["locale"]
                chunk_entries.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "locale": locale,
                    "text": chunk.text,
                    "metadata": dict(chunk.metadata, locale=locale, ordinal=ordinal),
                    "checksum": chunk.checksum,
                    "embedding_input": chunk.embedding_input,
                    "vector": None,
                    "cache_write": False,
                    "ordinal": ordinal,
                })

        cache_connection = await self._read_connection()
        try:
            for record in chunk_entries:
                cache_row = await self._cached_vector(
                    cache_connection,
                    str(record["checksum"]),
                    model_name,
                    dimension,
                )
                if cache_row is not None:
                    record["vector"] = validate_vector(
                        json.loads(cache_row["vector_json"]),
                        dimension,
                        label="cached embedding vector",
                    )
        finally:
            await cache_connection.close()

        missing_indexes = [
            index for index, record in enumerate(chunk_entries)
            if record["vector"] is None
        ]
        missing_texts = [
            str(chunk_entries[index]["embedding_input"])
            for index in missing_indexes
        ]

        if missing_texts:
            vectors = await embedding.embed(missing_texts)
            if len(vectors) != len(missing_texts):
                raise ValueError("embedding count does not match chunk count")
            for index, vector in zip(missing_indexes, vectors, strict=True):
                validated = validate_vector(
                    vector, dimension, label="embedding vector"
                )
                chunk_entries[index]["vector"] = validated
                chunk_entries[index]["cache_write"] = True
        return chunk_entries

    async def _cached_vector(
        self,
        connection: aiosqlite.Connection,
        checksum: str,
        model_name: str,
        dimension: int,
    ) -> sqlite3.Row | None:
        cursor = await connection.execute(
            """
            SELECT vector_json
            FROM embedding_cache
            WHERE checksum = ? AND model_name = ? AND embedding_dimension = ?
            """,
            (checksum, model_name, dimension),
        )
        return await cursor.fetchone()

    async def _read_connection(self) -> aiosqlite.Connection:
        if self._read_only:
            conn = await aiosqlite.connect(
                self._readonly_uri(), uri=True, isolation_level=None
            )
        else:
            conn = await aiosqlite.connect(self._path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        await conn.execute("PRAGMA query_only = ON")
        return conn

    def _readonly_uri(self) -> str:
        # Live inspection must see the writer's WAL. Immutable mode is retained
        # for existing snapshot/readiness users that require no sidecar creation.
        suffix = '' if self._live_readonly else '&immutable=1'
        return f"{self._path.resolve().as_uri()}?mode=ro{suffix}"


class SqliteHybridKnowledge:
    def __init__(
        self,
        index: SqliteKnowledgeIndex,
        *,
        vector_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        final_k: int = 5,
        threshold: float = 0.0,
    ) -> None:
        self._index = index
        self._vector_candidate_k = vector_candidate_k
        self._bm25_candidate_k = bm25_candidate_k
        self._final_k = final_k
        self._threshold = threshold

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        if query.query_vector is None:
            raise ValueError("query vector is required")
        if not query.embedding_model:
            raise ValueError("embedding model is required")
        if query.embedding_dimension is None:
            raise ValueError("embedding dimension is required")
        chunks = await self._index.search_vector(
            query=query.query_text,
            query_vector=query.query_vector,
            embedding_model=query.embedding_model,
            embedding_dimension=query.embedding_dimension,
            config=RagSearchConfig(
                locale=query.locale or "en",
                candidate_k=max(self._vector_candidate_k, self._bm25_candidate_k),
                vector_candidate_k=self._vector_candidate_k,
                bm25_candidate_k=self._bm25_candidate_k,
                final_k=min(query.top_k, self._final_k),
                threshold=self._threshold,
                allowed_document_ids=query.allowed_document_ids,
            ),
        )
        return [
            replace(
                chunk,
                metadata={
                    key: value
                    for key, value in chunk.metadata.items()
                    if key not in {"embedding", "source_path", "source_uri"}
                },
            )
            for chunk in chunks
        ]


def load_index_directory(
    knowledge_dir: str | Path,
    *,
    index_root: str | Path | None = None,
) -> Path:
    root = Path(index_root) if index_root is not None else Path.cwd() / ".rag-index"
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / _DEFAULT_INDEX_NAME
    return index_path


def default_index_path() -> Path:
    return load_index_directory(Path.cwd() / ".rag-index")


async def import_directory(
    knowledge_dir: str | Path,
    embedding,
    *,
    index_path: str | Path | None = None,
    locale: str | None = None,
) -> GenerationRecord:
    index = SqliteKnowledgeIndex(index_path or default_index_path())
    await index.open()
    try:
        return await index.import_directory(knowledge_dir, embedding, locale=locale)
    finally:
        await index.close()


def _load_document(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    locale = _locale_from_path(path)
    title = _title_from_text(path, text)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    relative = path.relative_to(root).as_posix()
    document_id = "doc-" + hashlib.sha256(
        f"{relative}:{checksum}:{title}".encode("utf-8")
    ).hexdigest()[:16]
    created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return Document(
        document_id=document_id,
        source_uri=str(path.resolve()),
        title=title,
        text=text,
        content_type="text/markdown" if path.suffix.lower() == ".md" else "text/plain",
        checksum=checksum,
        created_at=created_at,
        metadata={"locale": locale, "source_path": relative},
    )


def _title_from_text(path: Path, text: str) -> str:
    if path.suffix.lower() == ".md":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or _stem_without_locale(path)
    return _stem_without_locale(path)


def _stem_without_locale(path: Path) -> str:
    stem = path.stem
    parts = stem.split(".")
    if len(parts) >= 2 and _LOCALE_RE.match(parts[-1]):
        parts = parts[:-1]
    return ".".join(parts) if parts else stem


def _locale_from_path(path: Path) -> str:
    stem = path.stem
    parts = stem.split(".")
    if len(parts) >= 2 and _LOCALE_RE.match(parts[-1]):
        return normalize_locale(parts[-1])
    return "en"


def _override_locale(document: Document, locale: str) -> Document:
    metadata = dict(document.metadata)
    metadata["locale"] = normalize_locale(locale)
    return Document(
        document_id=document.document_id,
        source_uri=document.source_uri,
        title=document.title,
        text=document.text,
        content_type=document.content_type,
        checksum=document.checksum,
        created_at=document.created_at,
        metadata=metadata,
    )


def _embedding_metadata(embedding) -> tuple[str, int]:
    model_name = getattr(embedding, "model_name", None)
    dimension = getattr(embedding, "dimension", None)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("embedding metadata requires non-empty model_name")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("embedding metadata requires positive dimension")
    return model_name.strip(), dimension


def _row_to_retrieved_chunk(row: sqlite3.Row) -> RetrievedChunk:
    metadata = json.loads(row["metadata_json"])
    metadata["embedding"] = json.loads(row["vector_json"])
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        locale=row["locale"],
        text=row["text"],
        vector_rank=0,
        keyword_rank=0,
        fused_score=0.0,
        metadata=metadata,
    )


def _retrieve_chunks(
    *,
    rows: Sequence[sqlite3.Row],
    query: str,
    query_vector: Sequence[float],
    config: RagSearchConfig,
) -> list[RetrievedChunk]:
    # Scope before either candidate pool is ranked/truncated. Post-filtering a
    # global top-k loses valid evidence that ranked below unrelated documents.
    allowed = (None if config.allowed_document_ids is None
               else frozenset(config.allowed_document_ids))
    chunks = [_row_to_retrieved_chunk(row) for row in rows
              if allowed is None or row['document_id'] in allowed]
    dimension = len(query_vector)
    chunks = [
        replace(
            chunk,
            metadata=dict(
                chunk.metadata,
                embedding=validate_vector(
                    chunk.metadata.get("embedding"),
                    dimension,
                    label="stored embedding vector",
                ),
            ),
        )
        for chunk in chunks
    ]
    if not chunks:
        return []
    return HybridRetriever().search(
        chunks=chunks,
        query_vector=query_vector,
        query_text=query,
        locale=config.locale,
        candidate_k=config.candidate_k,
        final_k=config.final_k,
        threshold=config.threshold,
        vector_candidate_k=config.effective_vector_candidate_k,
        bm25_candidate_k=config.effective_bm25_candidate_k,
    )


def _generation_id() -> str:
    raw = f"{datetime.now(UTC).isoformat()}:{Path.cwd()}"
    return "gen-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
    generation_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    title TEXT NOT NULL,
    locale TEXT NOT NULL,
    checksum TEXT NOT NULL,
    content_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (generation_id, document_id),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    generation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    locale TEXT NOT NULL,
    text TEXT NOT NULL,
    checksum TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (generation_id, chunk_id),
    FOREIGN KEY (generation_id, document_id) REFERENCES documents(generation_id, document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    checksum TEXT NOT NULL,
    model_name TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY (checksum, model_name, embedding_dimension)
);

CREATE INDEX IF NOT EXISTS idx_generations_active_created_at
    ON generations(active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_generation_ordinal
    ON chunks(generation_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_lookup
    ON embedding_cache(checksum, model_name, embedding_dimension);
"""
