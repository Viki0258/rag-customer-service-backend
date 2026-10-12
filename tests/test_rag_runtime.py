from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from whatsapp_support.domain import (
    EventType,
    IdentityProof,
    IdentityResult,
    InboundEvent,
    RagAnswerResult,
    RetrievalQuery,
    RetrievedChunk,
    StructuredModelResult,
    TemplateDefinition,
)
from whatsapp_support.fakes import (
    FakeBusinessTools,
    FakeConversationMemory,
    FakeEmbedding,
    FakeHandoff,
    FakeHybridKnowledge,
    FakeIdentity,
    FakeModel,
    FakeTemplateCatalog,
    InMemoryConversationOutboxTransaction,
    InMemoryConversationStore,
    InMemoryOutboxStore,
)
from whatsapp_support.ecommerce_tools import FakeEcommerceTools
from whatsapp_support.rag import RagAnswerService
from whatsapp_support.rag_index import SqliteHybridKnowledge, SqliteKnowledgeIndex
from whatsapp_support.worker import ConversationWorker


NOW = datetime(2026, 9, 5, 10, tzinfo=UTC)


class RecordingRagService:
    def __init__(self, *, status: str = "reply", reason: str = "grounded_answer") -> None:
        self.status = status
        self.reason = reason
        self.calls: list[tuple[InboundEvent, str, str]] = []

    async def answer(
        self, event: InboundEvent, locale: str, trace_id: str
    ) -> RagAnswerResult:
        self.calls.append((event, locale, trace_id))
        return RagAnswerResult(
            status=self.status,
            reply=(
                "The backpack shell is recycled nylon."
                if self.status == "reply"
                else None
            ),
            locale=locale,
            trace_id=trace_id,
            retrieved_chunk_ids=("chunk-1",) if self.status == "reply" else (),
            supported_chunk_ids=("chunk-1",) if self.status == "reply" else (),
            prompt_version="prompt-v1",
            model_name="answer-v1",
            reason=self.reason,
        )


def make_event(
    *,
    text: str,
    event_key: str = "event-1",
    occurred_at: datetime = NOW,
    identity_proof: IdentityProof | None = None,
) -> InboundEvent:
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id="recipient-1",
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=occurred_at,
        received_at=occurred_at,
        payload_digest=f"digest-{event_key}",
        text=text,
        identity_proof=identity_proof,
    )


@pytest.mark.asyncio
async def test_rag_service_passes_the_single_query_embedding_and_identity_to_retrieval() -> None:
    embedding = FakeEmbedding(model_name="embedding-v1", dimension=2)
    embedding.results_queue.append([[0.25, 0.75]])
    knowledge = FakeHybridKnowledge()
    knowledge.results_queue.append(
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                locale="en",
                text="The backpack shell is recycled nylon.",
                vector_rank=1,
                keyword_rank=1,
                fused_score=0.9,
            )
        ]
    )
    model = FakeModel(model_name="answer-v1")
    model.results_queue.append(
        StructuredModelResult(
            trace_id="trace-1",
            model_name="answer-v1",
            prompt_version="prompt-v1",
            content={
                "reply": "The backpack shell is recycled nylon.",
                "supported_evidence_ids": ["chunk-1"],
                "handoff": False,
                "reason": "grounded_answer",
            },
            raw_text="{}",
            created_at=NOW,
        )
    )
    service = RagAnswerService(
        embedding=embedding,
        knowledge=knowledge,
        model=model,
        memory=FakeConversationMemory(),
        prompt_version="prompt-v1",
        retrieval_top_k=5,
        evidence_threshold=0.1,
        max_retrieved_chunks=3,
        max_reply_chars=500,
        clock=lambda: NOW,
    )

    result = await service.answer(
        make_event(text="What is the backpack made from?"), "en", "trace-1"
    )

    assert result.status == "reply"
    assert embedding.calls == [["What is the backpack made from?"]]
    assert len(knowledge.calls) == 1
    query = knowledge.calls[0]
    assert query.query_vector == (0.25, 0.75)
    assert query.embedding_model == "embedding-v1"
    assert query.embedding_dimension == 2


@pytest.mark.asyncio
async def test_worker_routes_static_sku_description_through_rag_service() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    rag = RecordingRagService()
    worker = ConversationWorker(
        conversations,
        outbox,
        tools=None,
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        rag_answer=rag,
        clock=lambda: NOW,
    )
    event = make_event(text="What is product sku-1 made from?")

    result = await worker.process(event)

    assert result.status == "replied"
    assert len(rag.calls) == 1
    assert rag.calls[0][0] is event
    assert rag.calls[0][1] == "en"
    assert rag.calls[0][2] == result.trace_id
    command = (await outbox.records())[0].command
    assert command.text == "The backpack shell is recycled nylon."
    assert command.trace_id == result.trace_id
    assert result.retrieved_chunk_ids == ("chunk-1",)
    assert result.supported_chunk_ids == ("chunk-1",)
    assert result.prompt_version == "prompt-v1"
    assert result.model_name == "answer-v1"
    assert command.retrieved_chunk_ids == ("chunk-1",)
    assert command.supported_chunk_ids == ("chunk-1",)
    assert command.prompt_version == "prompt-v1"
    assert command.model_name == "answer-v1"


@pytest.mark.asyncio
async def test_durable_worker_consumes_same_rag_result_and_commits_audit_metadata() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox)
    rag = RecordingRagService()
    worker = ConversationWorker(
        conversations,
        outbox,
        tools=None,
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        rag_answer=rag,
        transaction=transaction,
        clock=lambda: NOW,
    )

    result = await worker.process(make_event(text="What material is this product?"))

    command = (await outbox.records())[0].command
    assert result.status == "replied"
    assert result.retrieved_chunk_ids == ("chunk-1",)
    assert command.supported_chunk_ids == ("chunk-1",)
    assert command.prompt_version == "prompt-v1"
    assert command.model_name == "answer-v1"


@pytest.mark.asyncio
async def test_mixed_sku_inventory_request_uses_realtime_tool_not_rag() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    tools = FakeEcommerceTools(clock=lambda: NOW)
    rag = RecordingRagService()
    worker = ConversationWorker(
        conversations,
        outbox,
        tools=tools,
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        rag_answer=rag,
        clock=lambda: NOW,
    )

    result = await worker.process(make_event(text="Is product sku-1 in stock?"))

    assert result.status == "replied"
    assert rag.calls == []
    command = (await outbox.records())[0].command
    assert command.text == "Everyday Backpack is in stock (7 available)."


@pytest.mark.parametrize(
    "text",
    ["What is your delivery policy?", "What is your return policy?"],
)
@pytest.mark.asyncio
async def test_static_delivery_and_return_policies_use_rag(text: str) -> None:
    rag = RecordingRagService()
    outbox = InMemoryOutboxStore()
    worker = ConversationWorker(
        InMemoryConversationStore(),
        outbox,
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        rag_answer=rag,
        clock=lambda: NOW,
    )

    result = await worker.process(make_event(text=text))

    assert result.status == "replied"
    assert len(rag.calls) == 1


@pytest.mark.parametrize(
    ("text", "scope", "expected_reply"),
    [
        (
            "Can I return order-1 under your return policy?",
            "return:read",
            "Order order-1 is eligible for return: Within the 30-day return window.",
        ),
        (
            "Tracking order-1 under the shipping policy",
            "shipping:read",
            "Order order-1 ships with Example Express. Tracking TRACK-1; status: in_transit.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mixed_personal_live_requests_use_authorized_tool_not_rag(
    text: str, scope: str, expected_reply: str
) -> None:
    identity = FakeIdentity()
    identity.results_queue.append(
        IdentityResult(
            verified=True,
            scopes=frozenset({scope}),
            subject_id="customer-1",
            status="verified",
            expires_at=NOW + timedelta(minutes=5),
        )
    )
    tools = FakeEcommerceTools(clock=lambda: NOW)
    rag = RecordingRagService()
    outbox = InMemoryOutboxStore()
    worker = ConversationWorker(
        InMemoryConversationStore(),
        outbox,
        tools=tools,
        identity=identity,
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        rag_answer=rag,
        clock=lambda: NOW,
    )

    result = await worker.process(
        make_event(
            text=text,
            identity_proof=IdentityProof(kind="otp", value="123456"),
        )
    )

    assert result.status == "replied"
    assert rag.calls == []
    command = (await outbox.records())[0].command
    assert command.text == expected_reply


@pytest.mark.asyncio
async def test_write_risk_request_never_calls_rag() -> None:
    rag = RecordingRagService()
    handoff = FakeHandoff()
    worker = ConversationWorker(
        InMemoryConversationStore(),
        InMemoryOutboxStore(),
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=handoff,
        rag_answer=rag,
        clock=lambda: NOW,
    )

    result = await worker.process(
        make_event(text="Refund product sku-1 and connect me to a human")
    )

    assert result.status == "handed_off"
    assert rag.calls == []


@pytest.mark.asyncio
async def test_rag_handoff_creates_handoff_and_no_outbox() -> None:
    outbox = InMemoryOutboxStore()
    handoff = FakeHandoff()
    rag = RecordingRagService(status="handoff", reason="insufficient_evidence")
    worker = ConversationWorker(
        InMemoryConversationStore(),
        outbox,
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=FakeTemplateCatalog(),
        handoff=handoff,
        rag_answer=rag,
        clock=lambda: NOW,
    )

    result = await worker.process(make_event(text="What material is this product?"))

    assert result.status == "handed_off"
    assert result.reason == "insufficient_evidence"
    assert len(handoff.calls) == 1
    assert await outbox.records() == []


@pytest.mark.asyncio
async def test_valid_rag_reply_uses_template_outside_customer_service_window() -> None:
    templates = FakeTemplateCatalog()
    templates.results_queue.append(
        TemplateDefinition(
            name="customer_service_reply",
            locale="en",
            category="utility",
            body_template="{{reply}}",
            variable_names=("reply",),
        )
    )
    outbox = InMemoryOutboxStore()
    worker = ConversationWorker(
        InMemoryConversationStore(),
        outbox,
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=None,
        model=None,
        templates=templates,
        handoff=FakeHandoff(),
        rag_answer=RecordingRagService(),
        clock=lambda: NOW,
    )

    result = await worker.process(
        make_event(
            text="What material is this product?",
            occurred_at=NOW - timedelta(days=2),
        )
    )

    command = (await outbox.records())[0].command
    assert result.status == "replied"
    assert command.text is None
    assert command.template_name == "customer_service_reply"
    assert command.template_variables == {
        "reply": "The backpack shell is recycled nylon."
    }


@pytest.mark.asyncio
async def test_sqlite_hybrid_port_consumes_supplied_vector_and_strips_transport_metadata(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "returns.en.md").write_text(
        "# Returns\n\nUnused products can be returned within 30 days.",
        encoding="utf-8",
    )

    class ImportEmbedding:
        model_name = "embedding-v1"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "knowledge.sqlite3")
    await index.open()
    try:
        await index.import_directory(knowledge_dir, ImportEmbedding())
        knowledge = SqliteHybridKnowledge(
            index,
            vector_candidate_k=1,
            bm25_candidate_k=2,
            final_k=2,
        )

        chunks = await knowledge.retrieve(
            RetrievalQuery(
                query_text="What is the return window?",
                conversation_id="conversation-1",
                top_k=2,
                locale="en",
                trace_id="trace-1",
                query_vector=(1.0, 0.0),
                embedding_model="embedding-v1",
                embedding_dimension=2,
            )
        )
    finally:
        await index.close()

    assert chunks
    assert all("embedding" not in chunk.metadata for chunk in chunks)
    assert all("source_path" not in chunk.metadata for chunk in chunks)


@pytest.mark.asyncio
async def test_sqlite_hybrid_port_rejects_active_generation_embedding_mismatch(
    tmp_path: Path,
) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.en.txt").write_text("Returns take 30 days.", encoding="utf-8")

    class ImportEmbedding:
        model_name = "embedding-v1"
        dimension = 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    index = SqliteKnowledgeIndex(tmp_path / "knowledge.sqlite3")
    await index.open()
    try:
        await index.import_directory(knowledge_dir, ImportEmbedding())
        knowledge = SqliteHybridKnowledge(index)
        with pytest.raises(ValueError, match="active generation embedding mismatch"):
            await knowledge.retrieve(
                RetrievalQuery(
                    query_text="return window",
                    conversation_id="conversation-1",
                    top_k=2,
                    locale="en",
                    query_vector=(1.0, 0.0),
                    embedding_model="embedding-v2",
                    embedding_dimension=2,
                )
            )
    finally:
        await index.close()
