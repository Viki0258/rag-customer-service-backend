import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import whatsapp_support.runtime as runtime_module
from whatsapp_support.config import DurableSettings, RagSettings
from whatsapp_support.domain import EventType, InboundEvent
from whatsapp_support.ecommerce_tools import FakeEcommerceTools
from whatsapp_support.meta_whatsapp import HttpResponse
from whatsapp_support.orchestration import ConversationOrchestrator
from whatsapp_support.rag_index import RagSearchConfig, SqliteKnowledgeIndex
from whatsapp_support.runtime import create_runtime_components, rag_index_main


def test_default_runtime_injects_fake_ecommerce_orchestrator() -> None:
    components = create_runtime_components(
        DurableSettings(database_url=None, redis_url=None)
    )

    assert isinstance(components.worker._orchestrator, ConversationOrchestrator)
    assert isinstance(
        components.worker._orchestrator.registry.get("get_product_info"),
        FakeEcommerceTools,
    )

    asyncio.run(components.aclose())


def test_fake_runtime_wires_deterministic_grounded_rag_without_openai() -> None:
    async def scenario() -> None:
        components = await runtime_module.create_runtime_components_async(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=RagSettings(
                mode="fake",
                embedding_model="fake-embedding-v1",
                response_model="fake-grounded-v1",
                embedding_dimension=2,
                prompt_version="prompt-runtime-v7",
            ),
        )
        event_time = datetime.now(UTC)
        event = InboundEvent(
            event_key="evt-rag",
            phone_number_id="phone-1",
            conversation_id="conversation-1",
            contact_id="contact-1",
            recipient_id="recipient-1",
            provider_message_id="wamid-rag",
            event_type=EventType.MESSAGE,
            occurred_at=event_time,
            received_at=event_time,
            payload_digest="digest-rag",
            text="What material is this product made from?",
        )
        try:
            result = await components.worker.process(event)
            records = await components.outbox.records()
        finally:
            await components.aclose()

        assert result.status == "replied"
        assert len(records) == 1
        assert records[0].command.text == (
            "Fake catalog evidence: the sample backpack uses recycled nylon."
        )
        assert components.worker._rag_answer is not None
        assert components.worker._rag_answer._prompt_version == "prompt-runtime-v7"

    asyncio.run(scenario())


def test_durable_storage_can_use_fake_rag_independently() -> None:
    async def scenario() -> None:
        components = await runtime_module.create_runtime_components_async(
            DurableSettings(
                database_url="sqlite+aiosqlite:///:memory:",
                redis_url="redis://localhost:6379/15",
            ),
            rag_settings=RagSettings(
                mode="fake",
                embedding_model="fake-embedding-v1",
                response_model="fake-grounded-v1",
                embedding_dimension=2,
            ),
        )
        try:
            assert components.mode == "durable"
            assert components.worker._rag_answer is not None
            assert components.worker._rag_answer._memory is components.database.memory
        finally:
            await components.aclose()

    asyncio.run(scenario())


def test_configured_index_path_connects_cli_import_to_durable_worker(
    tmp_path: Path, monkeypatch
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.en.txt").write_text(
        "Fake returns are accepted within 30 days.", encoding="utf-8"
    )
    index_path = tmp_path / "configured.sqlite3"
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_MODE", "fake")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_INDEX_PATH", str(index_path))
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION", "2")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-v1")

    assert rag_index_main([str(knowledge_dir)]) == 0

    async def scenario() -> None:
        index = SqliteKnowledgeIndex(index_path)
        await index.open()
        try:
            active = await index.active_generation()
        finally:
            await index.close()
        assert active is not None
        assert active.embedding_model == "embedding-v1"
        assert active.embedding_dimension == 2

        class NoNetworkTransport:
            async def post(self, *args, **kwargs):
                raise AssertionError("no API call expected")

            async def aclose(self) -> None:
                return None

        components = await runtime_module.create_runtime_components_async(
            DurableSettings(
                database_url="sqlite+aiosqlite:///:memory:",
                redis_url="redis://localhost:6379/15",
            ),
            rag_settings=RagSettings(
                mode="openai",
                index_path=str(index_path),
                openai_api_key="test-key",
                embedding_model="embedding-v1",
                response_model="answer-v1",
                embedding_dimension=2,
            ),
            rag_transport_factory=NoNetworkTransport,
        )
        try:
            configured_index = components.worker._rag_answer._knowledge._index
            assert configured_index._path == index_path
        finally:
            await components.aclose()

    asyncio.run(scenario())


def test_openai_rag_selection_is_independent_of_durable_storage(tmp_path: Path) -> None:
    class NoNetworkTransport:
        async def post(self, *args, **kwargs):
            raise AssertionError("no API call expected")

        async def aclose(self) -> None:
            return None

    components = create_runtime_components(
        DurableSettings(database_url=None, redis_url=None),
        rag_settings=RagSettings(
            mode="openai",
            index_path=str(tmp_path / "configured.sqlite3"),
            openai_api_key="test-key",
            embedding_model="embedding-v1",
            response_model="answer-v1",
            embedding_dimension=2,
        ),
        rag_transport_factory=NoNetworkTransport,
    )
    try:
        assert components.worker._rag_answer._knowledge.__class__.__name__ == (
            "SqliteHybridKnowledge"
        )
        assert components.worker._rag_answer._model.__class__.__name__ == (
            "OpenAIResponseAdapter"
        )
    finally:
        asyncio.run(components.aclose())


def test_openai_adapters_run_through_runtime_service_without_network(
    tmp_path: Path,
) -> None:
    class ImportEmbedding:
        model_name = "embedding-v1"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    class RecordingTransport:
        def __init__(self, response: HttpResponse) -> None:
            self.response = response
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.closed = False

        async def post(self, url, *, headers, json_body, timeout):
            del headers, timeout
            self.calls.append((url, dict(json_body)))
            return self.response

        async def aclose(self) -> None:
            self.closed = True

    index_path = tmp_path / "configured.sqlite3"
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "catalog.en.txt").write_text(
        "The sample backpack uses recycled nylon.", encoding="utf-8"
    )

    async def scenario() -> None:
        index = SqliteKnowledgeIndex(index_path)
        await index.open()
        try:
            await index.import_directory(knowledge_dir, ImportEmbedding())
            imported_chunks = await index.search_vector(
                query="What material is the product made from?",
                query_vector=[1.0, 0.0],
                embedding_model="embedding-v1",
                embedding_dimension=2,
                config=RagSearchConfig(locale="en"),
            )
        finally:
            await index.close()

        embedding_transport = RecordingTransport(
            HttpResponse(200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        )
        model_reply = json.dumps(
            {
                "reply": "The sample backpack uses recycled nylon.",
                "supported_evidence_ids": [imported_chunks[0].chunk_id],
                "handoff": False,
                "reason": "grounded_answer",
            }
        )
        response_transport = RecordingTransport(
            HttpResponse(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": model_reply}],
                        }
                    ]
                },
            )
        )
        transports = iter((embedding_transport, response_transport))
        components = await runtime_module.create_runtime_components_async(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=RagSettings(
                mode="openai",
                index_path=str(index_path),
                openai_api_key="test-key",
                embedding_model="embedding-v1",
                response_model="answer-v1",
                embedding_dimension=2,
            ),
            rag_transport_factory=lambda: next(transports),
        )
        try:
            assert await components.readyz() is True
            event_time = datetime.now(UTC)
            event = InboundEvent(
                event_key="evt-adapter-service",
                phone_number_id="phone-1",
                conversation_id="conversation-1",
                contact_id="contact-1",
                recipient_id="recipient-1",
                provider_message_id="wamid-adapter-service",
                event_type=EventType.MESSAGE,
                occurred_at=event_time,
                received_at=event_time,
                payload_digest="digest-adapter-service",
                text="What material is the product made from?",
            )
            result = await components.worker.process(event)
        finally:
            await components.aclose()

        assert result.status == "replied"
        assert result.supported_chunk_ids == (imported_chunks[0].chunk_id,)
        assert len(embedding_transport.calls) == 1
        assert embedding_transport.calls[0][0].endswith("/v1/embeddings")
        assert embedding_transport.calls[0][1]["input"] == [event.text]
        assert len(response_transport.calls) == 1
        response_body = response_transport.calls[0][1]
        assert response_transport.calls[0][0].endswith("/v1/responses")
        assert "audit" not in response_body
        assert "conversation-1" not in json.dumps(response_body)
        assert "The sample backpack uses recycled nylon." in json.dumps(response_body)
        assert embedding_transport.closed is True
        assert response_transport.closed is True

    asyncio.run(scenario())


def test_openai_runtime_readiness_does_not_initialize_or_mutate_index(
    tmp_path: Path,
) -> None:
    class NoNetworkTransport:
        async def post(self, *args, **kwargs):
            raise AssertionError("no API call expected")

        async def aclose(self) -> None:
            return None

    async def readiness(index_path: Path) -> bool:
        components = await runtime_module.create_runtime_components_async(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=RagSettings(
                mode="openai",
                index_path=str(index_path),
                openai_api_key="test-key",
                embedding_model="embedding-v1",
                response_model="answer-v1",
                embedding_dimension=2,
            ),
            rag_transport_factory=NoNetworkTransport,
        )
        try:
            return await components.readyz()
        finally:
            await components.aclose()

    missing_path = tmp_path / "missing" / "index.sqlite3"
    assert asyncio.run(readiness(missing_path)) is False
    assert not missing_path.parent.exists()

    uninitialized_path = tmp_path / "uninitialized.sqlite3"
    uninitialized_path.touch()
    before_bytes = uninitialized_path.read_bytes()
    before_files = sorted(path.name for path in tmp_path.iterdir())

    assert asyncio.run(readiness(uninitialized_path)) is False

    assert uninitialized_path.read_bytes() == before_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == before_files
    with sqlite3.connect(uninitialized_path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall() == []

    valid_path = tmp_path / "valid.sqlite3"
    knowledge_dir = tmp_path / "valid-knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.en.txt").write_text("Returns are easy.", encoding="utf-8")

    async def initialize_valid_index() -> None:
        class ImportEmbedding:
            model_name = "embedding-v1"
            dimension = 2

            async def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        index = SqliteKnowledgeIndex(valid_path)
        await index.open()
        try:
            await index.import_directory(knowledge_dir, ImportEmbedding())
        finally:
            await index.close()

    asyncio.run(initialize_valid_index())
    before_valid = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    assert asyncio.run(readiness(valid_path)) is True

    after_valid = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after_valid == before_valid


def test_fake_runtime_construction_failure_closes_earlier_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CloseableEmbedding:
        model_name = "fake-embedding"
        dimension = 1

        def __init__(self) -> None:
            self.closed = False

        async def embed(self, texts):
            return [[1.0] for _ in texts]

        async def aclose(self) -> None:
            self.closed = True

    embedding = CloseableEmbedding()

    class CloseableSettings(RagSettings):
        def create_embedding(self, *, transport_factory=None):
            del transport_factory
            return embedding

    def fail_worker(*args, **kwargs):
        raise RuntimeError("worker construction failed")

    monkeypatch.setattr(runtime_module, "_conversation_worker", fail_worker)

    with pytest.raises(RuntimeError, match="worker construction failed"):
        create_runtime_components(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=CloseableSettings(mode="fake"),
        )

    assert embedding.closed is True


def test_openai_runtime_model_construction_failure_closes_embedding() -> None:
    class CloseableEmbedding:
        model_name = "embedding-v1"
        dimension = 2

        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    embedding = CloseableEmbedding()

    class FailingModelSettings(RagSettings):
        def create_embedding(self, *, transport_factory=None):
            del transport_factory
            return embedding

        def create_model(self, *, transport_factory=None):
            del transport_factory
            raise RuntimeError("model construction failed")

    with pytest.raises(RuntimeError, match="model construction failed"):
        create_runtime_components(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=FailingModelSettings(
                mode="openai",
                index_path="unused.sqlite3",
                openai_api_key="test-key",
                embedding_model="embedding-v1",
                response_model="answer-v1",
                embedding_dimension=2,
            ),
        )

    assert embedding.closed is True


def test_durable_runtime_queue_construction_failure_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CloseableDatabase:
        def __init__(self, *args, **kwargs) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    database = CloseableDatabase()

    def fail_queue(*args, **kwargs):
        raise RuntimeError("queue construction failed")

    import whatsapp_support.postgres as postgres_module
    import whatsapp_support.redis_queue as redis_queue_module

    monkeypatch.setattr(postgres_module, "PostgresDatabase", lambda *a, **k: database)
    monkeypatch.setattr(redis_queue_module, "RedisStreamQueue", fail_queue)

    with pytest.raises(RuntimeError, match="queue construction failed"):
        create_runtime_components(
            DurableSettings(
                database_url="sqlite+aiosqlite:///:memory:",
                redis_url="redis://localhost:6379/15",
            ),
            rag_settings=RagSettings(mode="fake"),
        )

    assert database.closed is True


@pytest.mark.asyncio
async def test_runtime_construction_failure_cleans_up_inside_active_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SharedCloseableAdapter:
        model_name = "fake-embedding"
        dimension = 1

        def __init__(self) -> None:
            self.close_count = 0

        async def embed(self, texts):
            return [[1.0] for _ in texts]

        async def aclose(self) -> None:
            await asyncio.sleep(0)
            self.close_count += 1

    adapter = SharedCloseableAdapter()

    class CloseableSettings(RagSettings):
        def create_embedding(self, *, transport_factory=None):
            del transport_factory
            return adapter

        def create_model(self, *, transport_factory=None):
            del transport_factory
            return adapter

    def fail_worker(*args, **kwargs):
        raise RuntimeError("worker construction failed")

    monkeypatch.setattr(runtime_module, "_conversation_worker", fail_worker)

    with pytest.raises(RuntimeError, match="active event loop"):
        create_runtime_components(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=CloseableSettings(mode="fake"),
        )
    assert adapter.close_count == 0

    with pytest.raises(RuntimeError, match="worker construction failed"):
        await runtime_module.create_runtime_components_async(
            DurableSettings(database_url=None, redis_url=None),
            rag_settings=CloseableSettings(
                mode="openai",
                index_path="unused.sqlite3",
                openai_api_key="test-key",
                embedding_model="fake-embedding",
                response_model="fake-model",
                embedding_dimension=1,
            ),
        )

    assert adapter.close_count == 1
