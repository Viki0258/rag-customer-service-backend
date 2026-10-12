from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import (
    Conversation,
    EventType,
    Evidence,
    HandoffRequest,
    IdentityProof,
    InboundEvent,
    KnowledgeChunk,
    ModelRequest,
    OutboundCommand,
    OutboundMode,
    RetrievalQuery,
    StructuredModelResult,
    TemplateDefinition,
    ToolRequest,
)
from whatsapp_support.fakes import (
    ConcurrentUpdateError,
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
    InMemoryInboxStore,
    InMemoryOutboxStore,
    InMemoryQueue,
    QueueUnavailable,
)


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_event(
    *,
    event_key: str = "evt-1",
    conversation_id: str = "conv-1",
    occurred_at: datetime | None = None,
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    timestamp = occurred_at or utc_datetime(2026, 8, 22, 9, 0, 0)
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id=conversation_id,
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=timestamp,
        received_at=timestamp,
        payload_digest=f"digest-{event_key}",
        text="hello",
    )


def make_conversation(*, version: int = 0) -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        version=version,
    )


def make_command(
    *, dedup_key: str = "cmd-1", recipient_id: str = "recipient-1"
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conv-1",
        recipient_id=recipient_id,
        mode=OutboundMode.FREEFORM,
        text="hi there",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=f"trace-{dedup_key}",
    )


def test_inbox_store_keeps_first_duplicate_record() -> None:
    async def scenario() -> None:
        store = InMemoryInboxStore()
        first = make_event(event_key="evt-dup")
        duplicate = make_event(
            event_key="evt-dup",
            occurred_at=utc_datetime(2026, 8, 22, 9, 5, 0),
        )

        assert await store.record_once(first) is True
        assert await store.record_once(duplicate) is False
        assert await store.get("evt-dup") == first
        assert await store.list_unqueued() == [first]

    asyncio.run(scenario())


def test_inbox_store_mark_queued_removes_event_from_unqueued_list() -> None:
    async def scenario() -> None:
        store = InMemoryInboxStore()
        first = make_event(event_key="evt-1")
        second = make_event(event_key="evt-2")

        await store.record_once(first)
        await store.record_once(second)
        await store.mark_queued("evt-1")

        assert await store.list_unqueued() == [second]
        assert await store.get("evt-1") == first

    asyncio.run(scenario())


def test_queue_publish_failure_leaves_persisted_event_visible_for_relay() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue(fail_next_publish=True)
        event = make_event(event_key="evt-queued")

        assert await inbox.record_once(event) is True
        with pytest.raises(QueueUnavailable):
            await queue.publish(event)

        assert await inbox.list_unqueued() == [event]
        assert await queue.take() is None

    asyncio.run(scenario())


def test_queue_publish_failure_is_one_shot_and_next_publish_succeeds() -> None:
    async def scenario() -> None:
        queue = InMemoryQueue(fail_next_publish=True)
        first = make_event(event_key="evt-fail")
        second = make_event(event_key="evt-pass")

        with pytest.raises(QueueUnavailable):
            await queue.publish(first)

        await queue.publish(second)

        assert queue.fail_next_publish is False
        assert await queue.pending_delivery_ids() == ["delivery-1"]
        assert await queue.take() == ("delivery-1", second)

    asyncio.run(scenario())


def test_queue_take_is_non_destructive_until_acknowledged() -> None:
    async def scenario() -> None:
        queue = InMemoryQueue()
        first = make_event(event_key="evt-1")
        second = make_event(event_key="evt-2")

        await queue.publish(first)
        await queue.publish(second)

        assert await queue.pending_delivery_ids() == ["delivery-1", "delivery-2"]
        assert await queue.take() == ("delivery-1", first)
        assert await queue.take() == ("delivery-1", first)

        await queue.acknowledge("delivery-1")

        assert await queue.pending_delivery_ids() == ["delivery-2"]
        assert await queue.take() == ("delivery-2", second)

    asyncio.run(scenario())


def test_inbox_store_processed_event_does_not_reappear_for_relay() -> None:
    async def scenario() -> None:
        store = InMemoryInboxStore()
        event = make_event(event_key="evt-processed")

        assert await store.record_once(event) is True
        await store.mark_queued(event.event_key)
        await store.mark_processed(event.event_key)

        assert await store.list_unqueued() == []
        assert await store.get(event.event_key) == event

    asyncio.run(scenario())


def test_queue_receive_is_non_destructive_until_acknowledged() -> None:
    async def scenario() -> None:
        queue = InMemoryQueue()
        first = make_event(event_key="evt-1")
        second = make_event(event_key="evt-2")

        await queue.publish(first)
        await queue.publish(second)

        delivery = await queue.receive(consumer="worker-1")
        assert delivery is not None
        assert delivery.delivery_id == "delivery-1"
        assert delivery.event == first
        assert await queue.pending_delivery_ids() == ["delivery-1", "delivery-2"]

        again = await queue.receive(consumer="worker-1")
        assert again == delivery

        await queue.acknowledge(delivery.delivery_id)

        next_delivery = await queue.receive(consumer="worker-1")
        assert next_delivery is not None
        assert next_delivery.delivery_id == "delivery-2"
        assert next_delivery.event == second

    asyncio.run(scenario())


def test_queue_reclaim_is_empty_without_leases() -> None:
    async def scenario() -> None:
        queue = InMemoryQueue()
        await queue.publish(make_event(event_key="evt-1"))

        assert await queue.reclaim(consumer="worker-1", min_idle_ms=1000) == []

    asyncio.run(scenario())


def test_conversation_store_rejects_stale_save_after_atomic_event_update() -> None:
    async def scenario() -> None:
        store = InMemoryConversationStore()
        original = make_conversation(version=0)
        await store.save(original)

        updated, changed = await store.update_from_event(
            make_event(occurred_at=utc_datetime(2026, 8, 22, 10, 0, 0))
        )
        stale = original

        assert changed is True
        assert updated.version == 1
        with pytest.raises(ConcurrentUpdateError):
            await store.save(stale)

        assert await store.get("conv-1") == updated

    asyncio.run(scenario())


def test_outbox_enqueue_ignores_duplicate_dedup_key() -> None:
    async def scenario() -> None:
        store = InMemoryOutboxStore()
        first = make_command(dedup_key="same-key")
        duplicate = make_command(dedup_key="same-key")

        await store.enqueue(first)
        await store.enqueue(duplicate)

        records = await store.records()
        assert len(records) == 1
        assert records[0].command == first
        assert records[0].attempts == 0
        assert records[0].state == "PENDING"

    asyncio.run(scenario())


def test_outbox_retryable_failure_can_transition_to_sent() -> None:
    async def scenario() -> None:
        store = InMemoryOutboxStore()
        command = make_command(dedup_key="retry-once")

        await store.enqueue(command)
        claimed = await store.claim_pending()
        assert claimed is not None
        assert claimed.command == command
        assert claimed.attempts == 1
        assert claimed.state == "SENDING"

        await store.mark_retryable("retry-once", "provider timeout")
        retryable = (await store.records())[0]
        assert retryable.state == "RETRYABLE_FAILED"
        assert retryable.last_error == "provider timeout"

        reclaimed = await store.claim_pending()
        assert reclaimed is not None
        assert reclaimed.attempts == 2
        assert reclaimed.state == "SENDING"

        await store.mark_sent("retry-once", "wamid-123")
        sent = (await store.records())[0]
        assert sent.state == "SENT"
        assert sent.provider_message_id == "wamid-123"
        assert sent.attempts == 2

    asyncio.run(scenario())


def test_outbox_can_dead_letter_after_bounded_attempts() -> None:
    async def scenario() -> None:
        store = InMemoryOutboxStore()
        await store.enqueue(make_command(dedup_key="dead-letter"))

        for expected_attempt in (1, 2, 3):
            claimed = await store.claim_pending()
            assert claimed is not None
            assert claimed.attempts == expected_attempt
            if expected_attempt < 3:
                await store.mark_retryable("dead-letter", f"temporary-{expected_attempt}")
            else:
                await store.mark_dead_letter("dead-letter", "permanent failure")

        final_record = (await store.records())[0]
        assert final_record.state == "DEAD_LETTER"
        assert final_record.attempts == 3
        assert final_record.last_error == "permanent failure"
        assert await store.claim_pending() is None

    asyncio.run(scenario())


def test_service_fakes_capture_call_history() -> None:
    async def scenario() -> None:
        tools = FakeBusinessTools()
        identity = FakeIdentity()
        knowledge = FakeKnowledge()
        knowledge_store = FakeKnowledgeStore()
        handoff = FakeHandoff()
        whatsapp = FakeWhatsApp()

        tool_request = ToolRequest(
            request_id="tool-1",
            conversation_id="conv-1",
            tool_name="lookup-order",
            arguments={"order_id": "123"},
            trace_id="trace-tool",
            requested_at=utc_datetime(2026, 8, 22, 11, 0, 0),
        )
        proof = IdentityProof(kind="otp", value="123456")
        query = RetrievalQuery(
            query_text="return policy",
            conversation_id="conv-1",
            top_k=3,
            locale="en",
            trace_id="trace-search",
        )
        evidence = Evidence(
            evidence_id="ev-1",
            kind="policy",
            source="kb",
            collected_at=utc_datetime(2026, 8, 22, 11, 1, 0),
            detail="return policy",
        )
        knowledge.results_queue.append([evidence])
        chunk = KnowledgeChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            ordinal=0,
            text="policy text",
            embedding_input="policy text",
            checksum="sum-1",
        )
        handoff_request = HandoffRequest(
            request_id="handoff-1",
            conversation_id="conv-1",
            reason="needs human",
            requested_by="bot",
            requested_at=utc_datetime(2026, 8, 22, 11, 2, 0),
        )
        command = make_command(dedup_key="send-1")

        await tools.execute(tool_request)
        await identity.verify("conv-1", proof)
        assert await knowledge.search(query) == [evidence]
        await knowledge_store.upsert_chunks([chunk])
        await handoff.create(handoff_request)
        await whatsapp.send(command)

        assert tools.calls == [tool_request]
        assert identity.calls == [("conv-1", proof)]
        assert knowledge.calls == [query]
        assert knowledge_store.calls == [[chunk]]
        assert handoff.calls == [handoff_request]
        assert whatsapp.calls == [command]

    asyncio.run(scenario())


def test_embedding_fake_returns_configured_result_and_tracks_call() -> None:
    async def scenario() -> None:
        embedding = FakeEmbedding()
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        embedding.results_queue.append(vectors)

        result = await embedding.embed(["hello", "world"])

        assert result == vectors
        assert embedding.calls == [["hello", "world"]]
        assert isinstance(embedding.model_name, str)
        assert embedding.model_name
        assert isinstance(embedding.dimension, int)
        assert embedding.dimension > 0

    asyncio.run(scenario())


def test_model_fake_returns_configured_result_and_tracks_call() -> None:
    async def scenario() -> None:
        model = FakeModel()
        request = ModelRequest(
            request_id="model-1",
            conversation_id="conv-1",
            prompt_name="reply",
            prompt_version="v1",
            inputs={"text": "hello"},
            trace_id="trace-model",
            requested_at=utc_datetime(2026, 8, 22, 11, 3, 0),
        )
        configured = StructuredModelResult(
            trace_id="trace-model",
            model_name="fixture-model",
            prompt_version="v9",
            content={"reply": "ok"},
            raw_text='{"reply":"ok"}',
            created_at=utc_datetime(2026, 8, 22, 11, 4, 0),
            tokens_in=12,
            tokens_out=4,
        )
        model.results_queue.append(configured)

        result = await model.generate(request)

        assert result == configured
        assert model.calls == [request]

    asyncio.run(scenario())


def test_template_catalog_fake_returns_configured_template_and_tracks_call() -> None:
    async def scenario() -> None:
        catalog = FakeTemplateCatalog()
        template = TemplateDefinition(
            name="order_update",
            locale="en",
            category="utility",
            body_template="Order {{status}}",
            variable_names=("status",),
        )
        catalog.results_queue.append(template)

        result = await catalog.resolve("order_update", "en")

        assert result == template
        assert catalog.calls == [("order_update", "en")]

    asyncio.run(scenario())


def test_service_fake_errors_queue_raises_configured_error_and_preserves_call_history() -> None:
    async def scenario() -> None:
        tools = FakeBusinessTools()
        whatsapp = FakeWhatsApp()
        tool_request = ToolRequest(
            request_id="tool-error",
            conversation_id="conv-1",
            tool_name="lookup-order",
            arguments={"order_id": "999"},
            trace_id="trace-tool-error",
            requested_at=utc_datetime(2026, 8, 22, 11, 5, 0),
        )
        command = make_command(dedup_key="send-error")
        tool_error = RuntimeError("tool unavailable")
        send_error = RuntimeError("provider unavailable")
        tools.errors_queue.append(tool_error)
        whatsapp.errors_queue.append(send_error)

        with pytest.raises(RuntimeError, match="tool unavailable"):
            await tools.execute(tool_request)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await whatsapp.send(command)

        assert tools.calls == [tool_request]
        assert whatsapp.calls == [command]

    asyncio.run(scenario())
