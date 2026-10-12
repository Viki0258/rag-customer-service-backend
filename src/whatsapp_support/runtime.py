from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Awaitable, Callable, Literal, TypeVar
from urllib.parse import parse_qs, urlsplit

from whatsapp_support.config import DurableSettings, RagSettings, WhatsAppProviderSettings
from whatsapp_support.domain import ModelRequest, RetrievedChunk, StructuredModelResult
from whatsapp_support.fakes import (
    FakeBusinessTools,
    FakeEmbedding,
    FakeHandoff,
    FakeIdentity,
    FakeKnowledge,
    FakeKnowledgeStore,
    FakeModel,
    FakeTemplateCatalog,
    FakeWhatsApp,
    InMemoryConversationStore,
    InMemoryConversationOutboxTransaction,
    InMemoryInboxStore,
    InMemoryOutboxStore,
    InMemoryQueue,
)
from whatsapp_support.ecommerce_tools import EcommerceToolRegistry, FakeEcommerceTools
from whatsapp_support.gateway import (
    ChallengeRejected,
    InboxRelay,
    WebhookGateway,
    verify_challenge,
)
from whatsapp_support.indexer import KnowledgeIndexer
from whatsapp_support.memory import (
    AsyncInMemoryConversationMemory,
    InMemoryConversationMemory,
    RollingSummaryPolicy,
    conservative_structured_summary,
)
from whatsapp_support.orchestration import ConversationOrchestrator
from whatsapp_support.policy import IntentRouter, SafetyPolicy
from whatsapp_support.rag import RagAnswerService
from whatsapp_support.rag_index import (
    SqliteHybridKnowledge,
    SqliteKnowledgeIndex,
    import_directory as import_knowledge_directory,
)
from whatsapp_support.replies import ReplyComposer
from whatsapp_support.meta_webhook import MetaWebhookAdapter
from whatsapp_support.meta_whatsapp import (
    AsyncHttpTransport,
    MetaProviderError,
    MetaWhatsAppAdapter,
)
from whatsapp_support.ports import WhatsAppPort
from whatsapp_support.worker import ConversationWorker, OutboxDispatcher, OutboxReclaimer


logger = logging.getLogger(__name__)


SUPPORTED_ROLES = ("gateway", "worker", "indexer")
INTEGRATION_CHECK_COMMAND = "integration-check"
RAG_INDEX_COMMAND = "rag-index"
USAGE = (
    "Usage:\n"
    "  python -m whatsapp_support gateway [--port PORT]\n"
    "  python -m whatsapp_support worker [--port PORT]\n"
    "  python -m whatsapp_support indexer [--port PORT]\n"
    "  python -m whatsapp_support rag-index KNOWLEDGE_DIR\n"
    "  python -m whatsapp_support integration-check"
)


@dataclass
class RuntimeComponents:
    mode: Literal["fake", "durable"]
    inbox: object
    queue: object
    conversations: object
    outbox: object
    gateway: object
    whatsapp: WhatsAppPort
    worker: ConversationWorker
    outbox_dispatcher: OutboxDispatcher
    outbox_reclaimer: OutboxReclaimer
    inbox_relay: InboxRelay
    indexer: KnowledgeIndexer
    database: object | None = None
    event_processor: object | None = None
    verify_token: str = "local-dev-token"
    _ready_probe: Callable[[], Awaitable[None]] | None = None
    _extra_resources: tuple[object, ...] = ()
    _closed: bool = False

    async def readyz(self) -> bool:
        if self._ready_probe is None:
            return True
        try:
            await self._ready_probe()
        except Exception:
            return False
        return True

    async def readiness_error(self) -> str | None:
        if self._ready_probe is None:
            return None
        try:
            await self._ready_probe()
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_error: BaseException | None = None
        seen: set[int] = set()
        for resource in (
            *self._extra_resources,
            self.whatsapp,
            self.queue,
            self.database,
        ):
            if resource is None:
                continue
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            closer = getattr(resource, "aclose", None)
            if closer is None:
                closer = getattr(resource, "close", None)
            if closer is None:
                continue
            try:
                await closer()
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise close_error


_AwaitedT = TypeVar("_AwaitedT")


class _AsyncLoopRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="whatsapp-support-async-runtime",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()

    def run(self, awaitable: Awaitable[_AwaitedT]) -> _AwaitedT:
        if self._closed:
            raise RuntimeError("async runtime is closed")
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result()

    def close(self, application: object) -> None:
        if self._closed:
            return
        close_error: BaseException | None = None
        try:
            closer = getattr(application, "aclose", None)
            if closer is not None:
                self.run(closer())
        except BaseException as exc:
            close_error = exc
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("async runtime did not stop")
        if close_error is not None:
            raise close_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()


class _ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, role: str, application: object) -> None:
        super().__init__(address, _RequestHandler)
        self.role = role
        self.application = application
        self.verify_token = getattr(application, "verify_token", "local-dev-token")
        self._async_runner = _AsyncLoopRunner()

    def run_async(self, awaitable: Awaitable[_AwaitedT]) -> _AwaitedT:
        return self._async_runner.run(awaitable)

    def server_close(self) -> None:
        super().server_close()
        self._async_runner.close(self.application)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _ServiceHTTPServer

    def do_GET(self) -> None:
        path = urlsplit(self.path)
        if path.path == "/healthz":
            self._send_json(200, {"role": self.server.role, "status": "ok"})
            return
        if path.path == "/readyz":
            if self._is_ready():
                self._send_json(200, {"role": self.server.role, "status": "ready"})
            else:
                self._send_json(503, {"role": self.server.role, "status": "not_ready"})
            return
        if self.server.role == "gateway" and path.path == "/webhook":
            self._handle_challenge(parse_qs(path.query, keep_blank_values=True))
            return
        self._send_json(404, {"error_code": "not_found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path)
        if self.server.role != "gateway" or path.path != "/webhook":
            self._send_json(404, {"error_code": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error_code": "invalid_content_length"})
            return
        if content_length < 0:
            self._send_json(400, {"error_code": "invalid_content_length"})
            return
        raw_body = self.rfile.read(content_length)
        gateway = getattr(self.server.application, "gateway", self.server.application)
        if not hasattr(gateway, "receive"):
            self._send_json(503, {"error_code": "gateway_unavailable"})
            return
        try:
            result = self.server.run_async(
                gateway.receive(
                    raw_body,
                    self.headers.get("X-Hub-Signature-256"),
                )
            )
        except Exception:
            self._send_json(500, {"error_code": "gateway_failure"})
            return
        self._send_json(result.status_code, asdict(result))

    def _handle_challenge(self, query: dict[str, list[str]]) -> None:
        mode = _first(query, "hub.mode")
        token = _first(query, "hub.verify_token")
        challenge = _first(query, "hub.challenge")
        if mode is None or token is None or challenge is None:
            self._send_json(403, {"error_code": "challenge_rejected"})
            return
        try:
            verified_challenge = verify_challenge(
                mode,
                token,
                challenge,
                self.server.verify_token,
            )
        except ChallengeRejected:
            self._send_json(403, {"error_code": "challenge_rejected"})
            return
        self._send_text(200, verified_challenge)

    def _is_ready(self) -> bool:
        readyz = getattr(self.server.application, "readyz", None)
        if readyz is None:
            return True
        try:
            return bool(self.server.run_async(readyz()))
        except Exception:
            return False

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_text(self, status: int, value: str) -> None:
        self._send_bytes(status, value.encode("utf-8"), "text/plain; charset=utf-8")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(
    role: str,
    port: int = 0,
    application_factory: Callable[[str], object] | None = None,
) -> ThreadingHTTPServer:
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"unsupported role: {role}")
    application = (
        application_factory(role)
        if application_factory is not None
        else _default_application(role)
    )
    return _ServiceHTTPServer(("127.0.0.1", port), role, application)


def _default_application(role: str) -> object:
    return create_runtime_components(DurableSettings.from_env())


class _FakeGroundedKnowledge:
    async def retrieve(self, query) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="fake-sample-product",
                document_id="fake-sample-catalog",
                locale=query.locale or "en",
                text="Fake catalog evidence: the sample backpack uses recycled nylon.",
                vector_rank=1,
                keyword_rank=1,
                fused_score=1.0,
                metadata={"fixture": "runtime-fake"},
            )
        ]


class _FakeGroundedModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def generate(self, request: ModelRequest) -> StructuredModelResult:
        evidence = request.inputs.get("evidence", [])
        evidence_id = evidence[0]["chunk_id"]
        return StructuredModelResult(
            trace_id=request.trace_id,
            model_name=self.model_name,
            prompt_version=request.prompt_version,
            content={
                "reply": "Fake catalog evidence: the sample backpack uses recycled nylon.",
                "supported_evidence_ids": [evidence_id],
                "handoff": False,
                "reason": "grounded_answer",
            },
            raw_text="{}",
            created_at=datetime.now(UTC),
        )


def create_runtime_components(
    settings: DurableSettings,
    *,
    meta_transport_factory: Callable[[], AsyncHttpTransport] | None = None,
    rag_settings: RagSettings | None = None,
    rag_transport_factory: Callable[[], AsyncHttpTransport] | None = None,
) -> RuntimeComponents:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "create_runtime_components cannot run inside an active event loop; "
            "use await create_runtime_components_async(...)"
        )
    created_resources: list[object] = []
    try:
        return _construct_runtime_components(
            settings,
            meta_transport_factory=meta_transport_factory,
            rag_settings=rag_settings,
            rag_transport_factory=rag_transport_factory,
            created_resources=created_resources,
        )
    except BaseException:
        asyncio.run(_close_construction_resources(created_resources))
        raise


async def create_runtime_components_async(
    settings: DurableSettings,
    *,
    meta_transport_factory: Callable[[], AsyncHttpTransport] | None = None,
    rag_settings: RagSettings | None = None,
    rag_transport_factory: Callable[[], AsyncHttpTransport] | None = None,
) -> RuntimeComponents:
    created_resources: list[object] = []
    try:
        return _construct_runtime_components(
            settings,
            meta_transport_factory=meta_transport_factory,
            rag_settings=rag_settings,
            rag_transport_factory=rag_transport_factory,
            created_resources=created_resources,
        )
    except BaseException:
        await _close_construction_resources(created_resources)
        raise


def _construct_runtime_components(
    settings: DurableSettings,
    *,
    meta_transport_factory: Callable[[], AsyncHttpTransport] | None,
    rag_settings: RagSettings | None,
    rag_transport_factory: Callable[[], AsyncHttpTransport] | None,
    created_resources: list[object],
) -> RuntimeComponents:
    rag_settings = rag_settings or RagSettings.from_env()
    rag_settings.validate()
    if settings.enabled:
        return _create_durable_runtime(
            settings,
            meta_transport_factory,
            rag_settings,
            rag_transport_factory,
            created_resources,
        )
    return _create_fake_runtime(
        settings,
        meta_transport_factory,
        rag_settings,
        rag_transport_factory,
        created_resources,
    )


def _create_fake_runtime(
    settings: DurableSettings,
    meta_transport_factory: Callable[[], AsyncHttpTransport] | None,
    rag_settings: RagSettings,
    rag_transport_factory: Callable[[], AsyncHttpTransport] | None,
    created_resources: list[object],
) -> RuntimeComponents:
    inbox = InMemoryInboxStore()
    queue = InMemoryQueue()
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "local-dev-token")
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "local-app-secret")
    memory_store = InMemoryConversationMemory(
        summary_policy=RollingSummaryPolicy(
            max_recent_turns=rag_settings.summary_retention_turns,
            summarize_at_turns=rag_settings.summary_trigger_turns,
            summarize_fn=conservative_structured_summary,
        )
    )
    memory = AsyncInMemoryConversationMemory(memory_store)
    embedding = rag_settings.create_embedding(
        transport_factory=rag_transport_factory
    )
    created_resources.append(embedding)
    rag_index: SqliteKnowledgeIndex | None = None
    if rag_settings.mode == "fake":
        knowledge = _FakeGroundedKnowledge()
        model = _FakeGroundedModel(rag_settings.response_model)
    else:
        rag_index = SqliteKnowledgeIndex(rag_settings.index_path)
        created_resources.append(rag_index)
        knowledge = SqliteHybridKnowledge(
            rag_index,
            vector_candidate_k=rag_settings.vector_candidate_count,
            bm25_candidate_k=rag_settings.bm25_candidate_count,
            final_k=rag_settings.final_candidate_count,
            threshold=rag_settings.evidence_threshold,
        )
        model = rag_settings.create_model(transport_factory=rag_transport_factory)
        created_resources.append(model)
    rag_answer = RagAnswerService(
        embedding=embedding,
        knowledge=knowledge,
        model=model,
        memory=memory,
        prompt_version=rag_settings.prompt_version,
        retrieval_top_k=rag_settings.final_candidate_count,
        evidence_threshold=rag_settings.evidence_threshold,
        max_retrieved_chunks=rag_settings.final_candidate_count,
        max_reply_chars=rag_settings.reply_max_chars,
        recent_turn_limit=rag_settings.summary_retention_turns,
        clock=lambda: datetime.now(UTC),
    )
    transaction = InMemoryConversationOutboxTransaction(
        conversations, outbox, memory_store
    )
    worker = _conversation_worker(
        conversations,
        outbox,
        transaction=transaction,
        rag_answer=rag_answer,
        memory=memory,
    )
    gateway, whatsapp, provider_ready = _build_whatsapp_provider(
        settings.whatsapp,
        inbox=inbox,
        queue=queue,
        app_secret=app_secret,
        transport_factory=meta_transport_factory,
    )
    created_resources.append(whatsapp)
    knowledge_store = FakeKnowledgeStore()

    async def ready_probe() -> None:
        if rag_index is not None:
            index_path = Path(rag_settings.index_path)
            if not index_path.is_file():
                raise RuntimeError("knowledge_index_unavailable")
            await rag_index.open_readonly()
            if await rag_index.active_generation() is None:
                raise RuntimeError("knowledge_index_unavailable")
        await provider_ready()

    return RuntimeComponents(
        mode="fake",
        inbox=inbox,
        queue=queue,
        conversations=conversations,
        outbox=outbox,
        gateway=gateway,
        whatsapp=whatsapp,
        worker=worker,
        outbox_dispatcher=OutboxDispatcher(outbox, whatsapp),
        outbox_reclaimer=OutboxReclaimer(outbox, lambda: datetime.now(UTC)),
        inbox_relay=InboxRelay(inbox, queue),
        indexer=KnowledgeIndexer(embedding, knowledge_store),
        verify_token=verify_token,
        _ready_probe=ready_probe,
        _extra_resources=tuple(
            resource
            for resource in (rag_index, embedding, model)
            if resource is not None
        ),
    )


def _create_durable_runtime(
    settings: DurableSettings,
    meta_transport_factory: Callable[[], AsyncHttpTransport] | None,
    rag_settings: RagSettings,
    rag_transport_factory: Callable[[], AsyncHttpTransport] | None,
    created_resources: list[object],
) -> RuntimeComponents:
    assert settings.database_url is not None
    assert settings.redis_url is not None
    from whatsapp_support.durable_worker import DurableEventProcessor
    from whatsapp_support.postgres import PostgresDatabase
    from whatsapp_support.redis_queue import RedisStreamQueue

    database = PostgresDatabase(
        settings.database_url,
        summary_policy=RollingSummaryPolicy(
            max_recent_turns=rag_settings.summary_retention_turns,
            summarize_at_turns=rag_settings.summary_trigger_turns,
            summarize_fn=conservative_structured_summary,
        ),
    )
    created_resources.append(database)
    queue = RedisStreamQueue(
        settings.redis_url,
        settings.stream_name,
        settings.consumer_group,
    )
    created_resources.append(queue)
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "local-dev-token")
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "local-app-secret")
    embedding = rag_settings.create_embedding(
        transport_factory=rag_transport_factory
    )
    created_resources.append(embedding)
    rag_index: SqliteKnowledgeIndex | None = None
    if rag_settings.mode == "fake":
        knowledge = _FakeGroundedKnowledge()
        model = _FakeGroundedModel(rag_settings.response_model)
    else:
        rag_index = SqliteKnowledgeIndex(rag_settings.index_path)
        created_resources.append(rag_index)
        knowledge = SqliteHybridKnowledge(
            rag_index,
            vector_candidate_k=rag_settings.vector_candidate_count,
            bm25_candidate_k=rag_settings.bm25_candidate_count,
            final_k=rag_settings.final_candidate_count,
            threshold=rag_settings.evidence_threshold,
        )
        model = rag_settings.create_model(transport_factory=rag_transport_factory)
        created_resources.append(model)
    rag_answer = RagAnswerService(
        embedding=embedding,
        knowledge=knowledge,
        model=model,
        memory=database.memory,
        prompt_version=rag_settings.prompt_version,
        retrieval_top_k=rag_settings.final_candidate_count,
        evidence_threshold=rag_settings.evidence_threshold,
        max_retrieved_chunks=rag_settings.final_candidate_count,
        max_reply_chars=rag_settings.reply_max_chars,
        recent_turn_limit=rag_settings.summary_retention_turns,
        clock=lambda: datetime.now(UTC),
    )
    worker = _conversation_worker(
        database.conversations,
        database.outbox,
        transaction=database.transaction_store,
        rag_answer=rag_answer,
        memory=database.memory,
    )
    gateway, whatsapp, provider_ready = _build_whatsapp_provider(
        settings.whatsapp,
        inbox=database.inbox,
        queue=queue,
        app_secret=app_secret,
        transport_factory=meta_transport_factory,
    )
    created_resources.append(whatsapp)

    async def ready_probe() -> None:
        if rag_index is not None:
            index_path = Path(rag_settings.index_path)
            if not index_path.is_file():
                raise RuntimeError("knowledge_index_unavailable")
            await rag_index.open_readonly()
            active = await rag_index.active_generation()
            if active is None:
                raise RuntimeError("knowledge_index_unavailable")
        await database.create_schema()
        await queue.ensure_group()
        await provider_ready()

    return RuntimeComponents(
        mode="durable",
        inbox=database.inbox,
        queue=queue,
        conversations=database.conversations,
        outbox=database.outbox,
        gateway=gateway,
        whatsapp=whatsapp,
        worker=worker,
        outbox_dispatcher=OutboxDispatcher(
            database.outbox,
            whatsapp,
            lease_seconds=settings.lease_seconds,
            max_attempts=settings.max_attempts,
        ),
        outbox_reclaimer=OutboxReclaimer(
            database.outbox,
            lambda: datetime.now(UTC),
            max_attempts=settings.max_attempts,
        ),
        inbox_relay=InboxRelay(database.inbox, queue),
        indexer=KnowledgeIndexer(FakeEmbedding(), FakeKnowledgeStore()),
        database=database,
        event_processor=DurableEventProcessor(
            worker,
            queue,
            consumer="worker-1",
            reclaim_idle_ms=settings.lease_seconds * 1000,
        ),
        verify_token=verify_token,
        _ready_probe=ready_probe,
        _extra_resources=tuple(
            resource
            for resource in (rag_index, embedding, model)
            if resource is not None
        ),
    )


async def _close_construction_resources(resources: list[object]) -> None:
    seen: set[int] = set()
    for resource in reversed(resources):
        if id(resource) in seen:
            continue
        seen.add(id(resource))
        closer = getattr(resource, "aclose", None)
        if closer is None:
            closer = getattr(resource, "close", None)
        if closer is None:
            continue
        try:
            await closer()
        except BaseException:
            logger.warning(
                "runtime construction cleanup failed",
                extra={"resource_type": type(resource).__name__},
            )


async def _always_ready() -> None:
    return None


async def _raise_provider_not_ready(provider: WhatsAppProviderSettings) -> None:
    provider.validate()


class UnavailableWhatsApp:
    async def send(self, command) -> object:
        del command
        raise MetaProviderError("provider_not_configured", retryable=False)


def _build_whatsapp_provider(
    provider: WhatsAppProviderSettings,
    *,
    inbox: object,
    queue: object,
    app_secret: str,
    transport_factory: Callable[[], AsyncHttpTransport] | None,
) -> tuple[object, WhatsAppPort, Callable[[], Awaitable[None]]]:
    if provider.mode == "fake":
        fake = FakeWhatsApp()
        return (
            WebhookGateway(inbox, queue, app_secret),
            fake,
            _always_ready,
        )
    if provider.mode != "meta_cloud":
        provider.validate()
    if not provider.is_complete:
        return (
            MetaWebhookAdapter(inbox, queue, app_secret),
            UnavailableWhatsApp(),
            lambda: _raise_provider_not_ready(provider),
        )
    provider.validate()
    if transport_factory is None:
        from whatsapp_support.meta_whatsapp import HttpxTransport

        transport = HttpxTransport()
    else:
        transport = transport_factory()
    assert provider.meta_access_token is not None
    assert provider.meta_phone_number_id is not None
    assert provider.meta_graph_base_url is not None
    sender = MetaWhatsAppAdapter(
        provider.meta_access_token,
        provider.meta_phone_number_id,
        provider.meta_graph_base_url,
        transport,
    )
    return (
        MetaWebhookAdapter(inbox, queue, app_secret),
        sender,
        _always_ready,
    )


def _conversation_worker(
    conversations: object,
    outbox: object,
    *,
    transaction: object | None,
    rag_answer=None,
    memory=None,
) -> ConversationWorker:
    ecommerce = FakeEcommerceTools()
    orchestrator = ConversationOrchestrator(
        memory=memory,
        tools=ecommerce,
        identity=FakeIdentity(),
        router=IntentRouter(),
        policy=SafetyPolicy(),
        composer=ReplyComposer(),
        registry=EcommerceToolRegistry(
            {
                "get_product_info": ecommerce,
                "get_product_inventory": ecommerce,
                "get_order_status": ecommerce,
                "get_shipping_tracking": ecommerce,
                "check_return_eligibility": ecommerce,
            }
        ),
    )
    return ConversationWorker(
        conversations,
        outbox,
        FakeBusinessTools(),
        FakeIdentity(),
        FakeKnowledge(),
        FakeModel(),
        FakeTemplateCatalog(),
        FakeHandoff(),
        transaction=transaction,
        orchestrator=orchestrator,
        rag_answer=rag_answer,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        print(USAGE)
        print("Roles: gateway, worker, indexer, rag-index, integration-check")
        return 0

    role = args.pop(0)
    if role == INTEGRATION_CHECK_COMMAND:
        if args:
            print("integration-check does not accept arguments", file=sys.stderr)
            return 2
        return integration_check_main()
    if role == RAG_INDEX_COMMAND:
        return rag_index_main(args)
    if role not in SUPPORTED_ROLES:
        print(f"unsupported role: {role}", file=sys.stderr)
        return 2
    if "--help" in args or "-h" in args:
        print(USAGE)
        return 0

    port = 0
    if args:
        if len(args) != 2 or args[0] != "--port":
            print("expected optional --port PORT", file=sys.stderr)
            return 2
        try:
            port = int(args[1])
        except ValueError:
            print("port must be an integer", file=sys.stderr)
            return 2

    server = create_server(role, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def role_main(role: str) -> int:
    return main([role, *sys.argv[1:]])


def rag_index_main(args: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if args is None else args)
    if len(command_args) != 1:
        print("rag-index expects exactly one directory argument", file=sys.stderr)
        return 2
    knowledge_dir = command_args[0]
    try:
        rag_settings = RagSettings.from_env()
        generation = asyncio.run(_import_configured_knowledge(knowledge_dir, rag_settings))
    except Exception as exc:
        print(f"rag-index failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "rag-index import complete: "
        f"{generation.generation_id} ({generation.embedding_dimension}d)"
    )
    return 0


async def _import_configured_knowledge(
    knowledge_dir: str, rag_settings: RagSettings
):
    embedding = rag_settings.create_embedding()
    try:
        return await import_knowledge_directory(
            knowledge_dir,
            embedding,
            index_path=rag_settings.index_path,
        )
    finally:
        closer = getattr(embedding, "aclose", None)
        if closer is not None:
            await closer()


def integration_check_main() -> int:
    try:
        settings = DurableSettings.from_env()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    components = create_runtime_components(settings)
    error = asyncio.run(_readiness_error_and_close(components))
    if error is not None:
        if not settings.enabled:
            print(
                "WhatsApp provider readiness check failed: "
                f"{error}",
                file=sys.stderr,
            )
            return 1
        print(
            "durable runtime dependency check failed: "
            f"{_redact_configured_urls(error, settings)}",
            file=sys.stderr,
        )
        return 1
    if not settings.enabled:
        print("durable runtime disabled; no integration dependencies configured")
        return 0
    print("durable runtime ready: PostgreSQL schema and Redis stream group verified")
    return 0


async def _readiness_error_and_close(components: RuntimeComponents) -> str | None:
    try:
        return await components.readiness_error()
    finally:
        closer = getattr(components, "aclose", None)
        if closer is not None:
            await closer()


def _first(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    if not values:
        return None
    return values[0]


def _redact_configured_urls(message: str, settings: DurableSettings) -> str:
    redacted = message
    for value in _url_redaction_candidates(settings):
        if value:
            redacted = redacted.replace(value, "[redacted-url]")
    return _CREDENTIAL_URL_PATTERN.sub("[redacted-url]", redacted)


def _url_redaction_candidates(settings: DurableSettings) -> set[str]:
    candidates = {
        value
        for value in (settings.database_url, settings.redis_url)
        if value is not None
    }
    database_url = settings.database_url
    if database_url is not None:
        if database_url.startswith("postgresql://"):
            candidates.add(
                "postgresql+asyncpg://"
                + database_url.removeprefix("postgresql://")
            )
        if database_url.startswith("postgres://"):
            candidates.add(
                "postgresql+asyncpg://" + database_url.removeprefix("postgres://")
            )
        if database_url.startswith("postgresql+asyncpg://"):
            candidates.add(
                "postgresql://"
                + database_url.removeprefix("postgresql+asyncpg://")
            )
    return candidates


_CREDENTIAL_URL_PATTERN = re.compile(
    r"\b(?:postgresql\+asyncpg|postgresql|postgres|rediss?|redis\+ssl)://"
    r"[^\s;,)\]]+@[^\s;,)\]]+"
)
