from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from whatsapp_support.domain import RetrievedChunk
from whatsapp_support.rag_index import (
    KnowledgeFileLoader,
    RagSearchConfig,
    SqliteKnowledgeIndex,
)


def _write_fixture(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_reads_markdown_and_txt_and_uses_locale_suffix(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")
    _write_fixture(tmp_path, "policies.zh-CN.txt", "退货政策\n\n支持七天退货。")
    _write_fixture(tmp_path, "ignored.html", "<p>ignore me</p>")

    docs = KnowledgeFileLoader.load_directory(tmp_path)
    docs_again = KnowledgeFileLoader.load_directory(tmp_path)

    assert [doc.title for doc in docs] == ["FAQ", "policies"]
    assert [doc.content_type for doc in docs] == ["text/markdown", "text/plain"]
    assert [doc.metadata["locale"] for doc in docs] == ["en", "zh-CN"]
    assert [doc.document_id for doc in docs] == [doc.document_id for doc in docs_again]
    assert all(doc.source_uri.endswith((".md", ".txt")) for doc in docs)


@pytest.mark.asyncio
async def test_import_directory_is_atomic_and_reuses_existing_embeddings(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")
    _write_fixture(tmp_path, "policies.zh-CN.md", "# 政策\n\n支持七天退货。")

    class RecordingEmbedding:
        model_name = "embedding-a"
        dimension = 2

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    embedding = RecordingEmbedding()
    await index.open()
    try:
        first = await index.import_directory(tmp_path, embedding)
        active = await index.active_generation()
        second = await index.import_directory(tmp_path, embedding)
        active_again = await index.active_generation()
    finally:
        await index.close()

    assert first.generation_id != second.generation_id
    assert active.generation_id == first.generation_id
    assert active_again.generation_id == second.generation_id
    assert len(embedding.calls) == 1
    assert sqlite3.connect(tmp_path / "index.sqlite3").execute(
        "PRAGMA journal_mode"
    ).fetchone()[0].lower() == "wal"


@pytest.mark.asyncio
async def test_import_embedding_runs_before_sqlite_writer_transaction(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class BlockingEmbedding:
        model_name = "embedding-a"
        dimension = 2

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.started.set()
            await self.release.wait()
            return [[1.0, 0.0] for _ in texts]

    database_path = tmp_path / "index.sqlite3"
    index = SqliteKnowledgeIndex(database_path)
    embedding = BlockingEmbedding()
    await index.open()
    import_task = asyncio.create_task(index.import_directory(knowledge_dir, embedding))
    writer = sqlite3.connect(database_path, timeout=0)
    writer_acquired = False
    try:
        await embedding.started.wait()
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer_acquired = True
        except sqlite3.OperationalError:
            pass
    finally:
        if writer_acquired:
            writer.rollback()
        writer.close()
        embedding.release.set()
        await import_task
        await index.close()

    assert writer_acquired, "embedding await held the SQLite writer transaction"


@pytest.mark.asyncio
async def test_cancelled_import_leaves_no_writer_lock_or_inactive_generation(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class StableEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class BlockingEmbedding(StableEmbedding):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    database_path = tmp_path / "index.sqlite3"
    index = SqliteKnowledgeIndex(database_path)
    await index.open()
    try:
        initial = await index.import_directory(knowledge_dir, StableEmbedding())
        _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns changed.")
        embedding = BlockingEmbedding()
        import_task = asyncio.create_task(
            index.import_directory(knowledge_dir, embedding)
        )
        await embedding.started.wait()
        import_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await import_task

        with sqlite3.connect(database_path, timeout=0) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.rollback()
        active = await index.active_generation()
        generation_count = sqlite3.connect(database_path).execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0]
    finally:
        await index.close()

    assert active is not None
    assert active.generation_id == initial.generation_id
    assert generation_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_sql",
    ["BEGIN IMMEDIATE", "INSERT INTO documents"],
)
async def test_cancellation_inside_import_transaction_rolls_back_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_sql: str,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class Embedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    database_path = tmp_path / "index.sqlite3"
    index = SqliteKnowledgeIndex(database_path)
    await index.open()
    try:
        initial = await index.import_directory(knowledge_dir, Embedding())
        _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns changed.")
        real_execute = aiosqlite.Connection.execute

        async def cancel_during_document_write(
            connection: aiosqlite.Connection,
            sql: str,
            parameters=None,
        ):
            if cancel_sql in sql:
                if cancel_sql == "BEGIN IMMEDIATE":
                    await real_execute(connection, sql)
                raise asyncio.CancelledError
            if parameters is None:
                return await real_execute(connection, sql)
            return await real_execute(connection, sql, parameters)

        monkeypatch.setattr(
            aiosqlite.Connection,
            "execute",
            cancel_during_document_write,
        )
        with pytest.raises(asyncio.CancelledError):
            await index.import_directory(knowledge_dir, Embedding())
        monkeypatch.setattr(aiosqlite.Connection, "execute", real_execute)

        with sqlite3.connect(database_path, timeout=0) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.rollback()
        active = await index.active_generation()
        generation_count = sqlite3.connect(database_path).execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0]
    finally:
        await index.close()

    assert active is not None
    assert active.generation_id == initial.generation_id
    assert generation_count == 1


@pytest.mark.asyncio
async def test_import_directory_preserves_active_generation_on_failure(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class StableEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class FlakyEmbedding:
        model_name = "embedding-a"
        dimension = 2

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            raise RuntimeError("boom")

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    stable = StableEmbedding()
    embedding = FlakyEmbedding()
    await index.open()
    try:
        initial = await index.import_directory(tmp_path, stable)
        assert (await index.active_generation()).generation_id == initial.generation_id
        _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are still easy.")
        with pytest.raises(RuntimeError, match="boom"):
            await index.import_directory(tmp_path, embedding)
        active_after_failure = await index.active_generation()
        generation_count = sqlite3.connect(tmp_path / "index.sqlite3").execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0]
    finally:
        await index.close()

    assert embedding.calls == 1
    assert active_after_failure is not None
    assert active_after_failure.generation_id == initial.generation_id
    assert generation_count == 1


@pytest.mark.asyncio
async def test_import_directory_rebuilds_when_embedding_model_or_dimension_changes(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class ModelEmbedding:
        def __init__(self, model_name: str, dimension: int) -> None:
            self.model_name = model_name
            self.dimension = dimension
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[float(i + 1)] * self.dimension for i, _ in enumerate(texts)]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    first_embedding = ModelEmbedding("embedding-a", 2)
    second_embedding = ModelEmbedding("embedding-b", 3)
    await index.open()
    try:
        await index.import_directory(tmp_path, first_embedding)
        await index.import_directory(tmp_path, second_embedding)
    finally:
        await index.close()

    assert len(first_embedding.calls) == 1
    assert len(second_embedding.calls) == 1
    assert first_embedding.calls[0] == second_embedding.calls[0]


@pytest.mark.asyncio
async def test_import_directory_rejects_missing_embedding_metadata_without_changing_active_generation(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class StableEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class MissingMetadataEmbedding:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        initial = await index.import_directory(tmp_path, StableEmbedding())
        with pytest.raises(ValueError, match="embedding metadata"):
            await index.import_directory(tmp_path, MissingMetadataEmbedding())
        active_after_failure = await index.active_generation()
        generation_count = sqlite3.connect(tmp_path / "index.sqlite3").execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0]
    finally:
        await index.close()

    assert active_after_failure is not None
    assert active_after_failure.generation_id == initial.generation_id
    assert generation_count == 1


@pytest.mark.asyncio
async def test_import_directory_does_not_reuse_cache_when_same_instance_model_name_changes(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class MutableEmbedding:
        def __init__(self) -> None:
            self.model_name = "embedding-a"
            self.dimension = 2
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    embedding = MutableEmbedding()
    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        await index.import_directory(tmp_path, embedding)
        embedding.model_name = "embedding-b"
        await index.import_directory(tmp_path, embedding)
    finally:
        await index.close()

    assert embedding.calls == [
        ["# FAQ", "Returns are easy."],
        ["# FAQ", "Returns are easy."],
    ]


@pytest.mark.asyncio
async def test_import_directory_rolls_back_when_declared_dimension_does_not_match_vectors(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class StableEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class WrongDimensionEmbedding:
        model_name = "embedding-a"
        dimension = 3

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    invalid = WrongDimensionEmbedding()
    await index.open()
    try:
        initial = await index.import_directory(tmp_path, StableEmbedding())
        _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are still easy.")
        with pytest.raises(ValueError, match="embedding dimension mismatch"):
            await index.import_directory(tmp_path, invalid)
        active_after_failure = await index.active_generation()
        generation_count = sqlite3.connect(tmp_path / "index.sqlite3").execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0]
    finally:
        await index.close()

    assert invalid.calls == 1
    assert active_after_failure is not None
    assert active_after_failure.generation_id == initial.generation_id
    assert generation_count == 1


@pytest.mark.asyncio
async def test_import_directory_keeps_cache_separated_by_declared_dimension(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class MutableDimensionEmbedding:
        def __init__(self) -> None:
            self.model_name = "embedding-a"
            self.dimension = 2
            self.calls: list[tuple[int, list[str]]] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append((self.dimension, list(texts)))
            return [[float(index + 1)] * self.dimension for index, _ in enumerate(texts)]

    embedding = MutableDimensionEmbedding()
    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        first = await index.import_directory(tmp_path, embedding)
        embedding.dimension = 3
        second = await index.import_directory(tmp_path, embedding)
        embedding.dimension = 2
        third = await index.import_directory(tmp_path, embedding)
    finally:
        await index.close()

    assert first.embedding_dimension == 2
    assert second.embedding_dimension == 3
    assert third.embedding_dimension == 2
    assert embedding.calls == [
        (2, ["# FAQ", "Returns are easy."]),
        (3, ["# FAQ", "Returns are easy."]),
    ]


@pytest.mark.asyncio
async def test_search_embeds_before_pinning_active_generation_transaction(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nOld answer only.")

    class BlockingEmbedding:
        model_name = "embedding-a"
        dimension = 2

        def __init__(self, query_vector: list[float]) -> None:
            self.query_vector = query_vector
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: list[str]) -> list[list[float]]:
            if len(texts) == 1 and texts[0] == "old answer":
                self.started.set()
                await self.release.wait()
                return [self.query_vector]
            return [[1.0, 0.0] for _ in texts]

    class ImportEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        await index.import_directory(knowledge_dir, ImportEmbedding())
        search_embedding = BlockingEmbedding([1.0, 0.0])

        search_task = asyncio.create_task(
            index.search(
                "old answer",
                search_embedding,
                RagSearchConfig(locale="en", candidate_k=5, final_k=3, threshold=0.0),
            )
        )
        await search_embedding.started.wait()

        updated_dir = tmp_path / "knowledge-updated"
        updated_dir.mkdir()
        _write_fixture(updated_dir, "faq.en.md", "# FAQ\n\nNew answer only.")
        await index.import_directory(updated_dir, ImportEmbedding())
        search_embedding.release.set()
        results = await search_task
    finally:
        await index.close()

    assert results
    assert isinstance(results[0], RetrievedChunk)
    assert "New answer only." in results[0].text


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [True, 0.0],
        ["1.0", 0.0],
        [1 + 0j, 0.0],
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [float("-inf"), 0.0],
        [1.0],
    ],
)
@pytest.mark.asyncio
async def test_import_rejects_malformed_vectors(
    tmp_path: Path, invalid_vector: list[object]
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.txt", "Returns are easy.")

    class MalformedEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[object]]:
            return [invalid_vector for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        with pytest.raises(ValueError, match="embedding vector"):
            await index.import_directory(knowledge_dir, MalformedEmbedding())
        assert await index.active_generation() is None
    finally:
        await index.close()


@pytest.mark.parametrize(
    "invalid_vector",
    [[False, 0.0], [None, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]],
)
@pytest.mark.asyncio
async def test_query_rejects_malformed_vectors(
    tmp_path: Path, invalid_vector: list[object]
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.txt", "Returns are easy.")

    class ValidEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        await index.import_directory(knowledge_dir, ValidEmbedding())
        with pytest.raises(ValueError, match="query vector"):
            await index.search_vector(
                query="returns",
                query_vector=invalid_vector,
                embedding_model="embedding-a",
                embedding_dimension=2,
                config=RagSearchConfig(locale="en"),
            )
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_import_rejects_malformed_cached_vector(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.txt", "Returns are easy.")

    class ValidEmbedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        await index.import_directory(knowledge_dir, ValidEmbedding())
        await index._connection.execute(
            "UPDATE embedding_cache SET vector_json = ?", ("[NaN, 0.0]",)
        )
        await index._connection.commit()

        with pytest.raises(ValueError, match="cached embedding vector"):
            await index.import_directory(knowledge_dir, ValidEmbedding())
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_prune_old_generations_keeps_previous_generation(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easy.")

    class Embedding:
        model_name = "embedding-a"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "index.sqlite3")
    await index.open()
    try:
        await index.import_directory(tmp_path, Embedding())
        _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are still easy.")
        await index.import_directory(tmp_path, Embedding())
        _write_fixture(tmp_path, "faq.en.md", "# FAQ\n\nReturns are easiest.")
        third = await index.import_directory(tmp_path, Embedding())
        active = await index.active_generation()
        removed = await index.prune_old_generations(keep=2)
    finally:
        await index.close()

    generations = sqlite3.connect(tmp_path / "index.sqlite3").execute(
        "SELECT generation_id, active FROM generations ORDER BY created_at DESC"
    ).fetchall()

    assert removed == 0
    assert active is not None
    assert active.generation_id == third.generation_id
    assert len(generations) == 2
    assert generations[0][0] == third.generation_id
    assert generations[0][1] == 1
    assert sum(row[1] for row in generations) == 1


def test_rag_index_cli_imports_into_rag_index_directory(tmp_path: Path, monkeypatch):
    from whatsapp_support.__main__ import main

    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    _write_fixture(knowledge_dir, "faq.en.md", "# FAQ\n\nReturns are easy.")
    monkeypatch.chdir(tmp_path)

    assert main(["rag-index", str(knowledge_dir)]) == 0
    assert (tmp_path / ".rag-index").exists()
