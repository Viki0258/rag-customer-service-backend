from __future__ import annotations

from dataclasses import replace
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from whatsapp_support.domain import (
    AutomationState,
    CommitResult,
    Conversation,
    Evidence,
    EventType,
    HandoffRequest,
    HandoffResult,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    QueueDelivery,
    RagAnswerResult,
    StructuredModelResult,
    TemplateDefinition,
    ToolResult,
)
from whatsapp_support.durable_worker import DurableEventProcessor
from whatsapp_support.ecommerce_tools import FakeEcommerceTools
from whatsapp_support.fakes import (
    FakeBusinessTools,
    FakeHandoff,
    FakeIdentity,
    FakeKnowledge,
    FakeModel,
    FakeTemplateCatalog,
    InMemoryConversationOutboxTransaction,
    InMemoryConversationStore,
    InMemoryOutboxStore,
)
from whatsapp_support.worker import ConversationWorker


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


class StaticRagAnswer:
    async def answer(
        self, event: InboundEvent, locale: str, trace_id: str
    ) -> RagAnswerResult:
        return RagAnswerResult(
            status="reply",
            reply="Policy answer",
            locale=locale,
            trace_id=trace_id,
            retrieved_chunk_ids=("chunk-1",),
            supported_chunk_ids=("chunk-1",),
            prompt_version="prompt-v1",
            model_name="answer-v1",
            reason="grounded_answer",
        )


def make_event(
    *,
    event_key: str = "event-1",
    text: str = "Is product sku-1 in stock?",
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=NOW,
        received_at=NOW,
        payload_digest=f"digest-{event_key}",
        text=text,
    )


def make_delivery(
    *, delivery_id: str = "delivery-1", event: InboundEvent | None = None
) -> QueueDelivery:
    return QueueDelivery(delivery_id=delivery_id, event=event or make_event())


def make_command(
    *, dedup_key: str = "reply:event-1", recipient_id: str = "recipient-1"
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conversation-1",
        recipient_id=recipient_id,
        mode=OutboundMode.FREEFORM,
        text="ok",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=f"trace-{dedup_key}",
    )


class RecordingQueue:
    def __init__(self, *deliveries: QueueDelivery) -> None:
        self._ready = list(deliveries)
        self._pending: dict[str, QueueDelivery] = {}
        self.ack_order: list[str] = []
        self.log: list[str] = []

    async def receive(
        self, *, consumer: str, block_ms: int = 0
    ) -> QueueDelivery | None:
        del consumer, block_ms
        if not self._ready:
            return None
        delivery = self._ready.pop(0)
        self._pending[delivery.delivery_id] = delivery
        return delivery

    async def reclaim(self, *, consumer: str, min_idle_ms: int) -> list[QueueDelivery]:
        del consumer, min_idle_ms
        deliveries = list(self._pending.values())
        self._pending.clear()
        for delivery in deliveries:
            self._pending[delivery.delivery_id] = delivery
        return deliveries

    async def acknowledge(self, delivery_id: str) -> None:
        self.log.append(f"ack:{delivery_id}")
        self.ack_order.append(delivery_id)
        self._pending.pop(delivery_id, None)

    def pending_delivery_ids(self) -> list[str]:
        return list(self._pending)


class RecordingTransaction:
    def __init__(
        self,
        *,
        fail: bool = False,
        result: CommitResult | None = None,
    ) -> None:
        self.fail = fail
        self.result = result
        self.committed = False
        self.conversations: dict[str, Conversation] = {}
        self.commands: list[OutboundCommand] = []
        self.processed_event_keys: set[str] = set()
        self.pending_handoffs: dict[str, tuple[str, HandoffRequest]] = {}
        self.completed_handoffs: set[str] = set()
        self.log: list[str] = []

    async def commit_event(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        command: OutboundCommand | None,
        handoff_request: HandoffRequest | None = None,
    ) -> CommitResult:
        self.log.append(f"commit:{event.event_key}")
        if self.fail:
            raise RuntimeError("injected transaction failure")
        if self.result is not None:
            return self.result
        if event.event_key in self.processed_event_keys:
            pending = self.pending_handoffs.get(event.event_key)
            return CommitResult(
                status="duplicate",
                reason="event_already_processed",
                handoff_request=None if pending is None else pending[1],
            )

        current = self.conversations.get(event.conversation_id)
        current_version = 0 if current is None else current.version
        if current_version != expected_version:
            return CommitResult(status="stale", reason="conversation_version_mismatch")

        self.conversations[event.conversation_id] = conversation
        if command is not None and not any(
            existing.dedup_key == command.dedup_key for existing in self.commands
        ):
            self.commands.append(command)
        self.processed_event_keys.add(event.event_key)
        if handoff_request is not None:
            self.pending_handoffs[event.event_key] = (
                handoff_request.request_id,
                handoff_request,
            )
        self.committed = True
        return CommitResult(status="committed", handoff_request=handoff_request)

    async def mark_handoff_complete(self, request_id: str) -> None:
        self.completed_handoffs.add(request_id)
        for event_key, (pending_request_id, _) in list(self.pending_handoffs.items()):
            if pending_request_id == request_id:
                self.pending_handoffs.pop(event_key)

    async def get(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)


class RecordingHandoff:
    def __init__(self, log: list[str] | None = None, *, fail: bool = False) -> None:
        self.log = log if log is not None else []
        self.fail = fail
        self.calls: list[HandoffRequest] = []

    async def create(self, request: HandoffRequest) -> HandoffResult:
        self.log.append(f"handoff:{request.request_id}")
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("handoff provider failed")
        return HandoffResult(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            accepted=True,
            queue_name="human-handoff",
            assigned_agent_id=None,
            accepted_at=NOW,
        )


def make_worker(
    *,
    transaction: RecordingTransaction,
    tools: FakeBusinessTools | None = None,
    handoff: RecordingHandoff | FakeHandoff | None = None,
) -> ConversationWorker:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    tools = tools or FakeEcommerceTools(clock=lambda: NOW)
    identity = FakeIdentity()
    knowledge = FakeKnowledge()
    model = FakeModel()
    templates = FakeTemplateCatalog()
    handoff = handoff or FakeHandoff()
    return ConversationWorker(
        conversations=conversations,
        outbox=outbox,
        tools=tools,
        identity=identity,
        knowledge=knowledge,
        model=model,
        templates=templates,
        handoff=handoff,
        transaction=transaction,
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_processor_acks_only_after_transaction_commit() -> None:
    transaction = RecordingTransaction()
    queue = RecordingQueue(make_delivery())
    transaction.log = queue.log
    processor = DurableEventProcessor(
        make_worker(transaction=transaction),
        queue,
        consumer="worker-a",
    )

    result = await processor.process_once()

    assert result is not None
    assert transaction.committed is True
    assert queue.ack_order == ["delivery-1"]
    assert queue.log == ["commit:event-1", "ack:delivery-1"]


@pytest.mark.asyncio
async def test_processor_leaves_delivery_unacknowledged_when_transaction_raises() -> None:
    transaction = RecordingTransaction(fail=True)
    queue = RecordingQueue(make_delivery())
    processor = DurableEventProcessor(
        make_worker(transaction=transaction),
        queue,
        consumer="worker-a",
    )

    with pytest.raises(RuntimeError, match="injected transaction failure"):
        await processor.process_once()

    assert queue.ack_order == []
    assert queue.pending_delivery_ids() == ["delivery-1"]
    assert [item.delivery_id for item in await queue.reclaim(
        consumer="worker-b",
        min_idle_ms=0,
    )] == ["delivery-1"]


@pytest.mark.asyncio
async def test_processor_hands_off_and_acks_when_tool_raises() -> None:
    transaction = RecordingTransaction()
    tools = FakeBusinessTools()
    tools.errors_queue.append(RuntimeError("tool is down"))
    queue = RecordingQueue(make_delivery())
    processor = DurableEventProcessor(
        make_worker(transaction=transaction, tools=tools),
        queue,
        consumer="worker-a",
    )

    result = await processor.process_once()

    assert transaction.commands == []
    assert result is not None
    assert result.status == "handed_off"
    assert result.reason == "tool_failed"
    assert queue.ack_order == ["delivery-1"]
    assert queue.pending_delivery_ids() == []


@pytest.mark.asyncio
async def test_duplicate_delivery_with_same_event_key_does_not_create_second_command() -> None:
    event = make_event(event_key="event-dup", recipient_id="wa-durable-42")
    transaction = RecordingTransaction()
    queue = RecordingQueue(
        make_delivery(delivery_id="delivery-1", event=event),
        make_delivery(delivery_id="delivery-2", event=replace(event)),
    )
    processor = DurableEventProcessor(
        make_worker(transaction=transaction),
        queue,
        consumer="worker-a",
    )

    first = await processor.process_once()
    second = await processor.process_once()

    assert first is not None
    assert first.outbox_created is True
    assert second is not None
    assert second.outbox_created is False
    assert second.status == "stale_or_duplicate"
    assert queue.ack_order == ["delivery-1", "delivery-2"]
    assert [command.dedup_key for command in transaction.commands] == [
        "reply:event-dup"
    ]
    assert transaction.commands[0].recipient_id == event.recipient_id


@pytest.mark.asyncio
async def test_durable_worker_without_orchestrator_composes_ecommerce_output_safely() -> None:
    transaction = RecordingTransaction()
    tools = FakeBusinessTools()
    tools.results_queue.append(
        ToolResult(
            request_id="unused",
            tool_name="get_product_inventory",
            success=True,
            output_text="INTERNAL RAW TOOL OUTPUT",
            output_data={
                "product_id": "sku-1",
                "name": "Everyday Backpack",
                "available": True,
                "quantity": 7,
            },
            completed_at=NOW,
        )
    )
    worker = make_worker(transaction=transaction, tools=tools)

    result = await worker.process(make_event(text="inventory sku-1"))

    assert result.status == "replied"
    assert transaction.commands[0].text == (
        "Everyday Backpack is in stock (7 available)."
    )
    assert "INTERNAL RAW TOOL OUTPUT" not in transaction.commands[0].text


@pytest.mark.asyncio
async def test_durable_expired_window_hands_off_on_template_locale_mismatch() -> None:
    transaction = RecordingTransaction()
    knowledge = FakeKnowledge()
    knowledge.results_queue.append(
        [
            Evidence(
                evidence_id="policy-1",
                kind="policy",
                source="fixture",
                collected_at=NOW,
                detail="Policy evidence",
            )
        ]
    )
    model = FakeModel()
    model.results_queue.append(
        StructuredModelResult(
            trace_id="model-trace",
            model_name="fixture-model",
            prompt_version="v1",
            content={"reply": "Policy answer"},
            raw_text='{"reply":"Policy answer"}',
            created_at=NOW,
        )
    )
    templates = FakeTemplateCatalog()
    templates.results_queue.append(
        TemplateDefinition(
            name="customer_service_reply",
            locale="zh-CN",
            category="utility",
            body_template="{{reply}}",
            variable_names=("reply",),
        )
    )
    worker = ConversationWorker(
        conversations=InMemoryConversationStore(),
        outbox=InMemoryOutboxStore(),
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=knowledge,
        model=model,
        templates=templates,
        handoff=FakeHandoff(),
        transaction=transaction,
        clock=lambda: NOW + timedelta(hours=25),
        rag_answer=StaticRagAnswer(),
    )

    result = await worker.process(make_event(text="What is your return policy?"))

    assert result.status == "handed_off"
    assert result.reason == "approved_template_locale_mismatch"
    assert transaction.commands == []


@pytest.mark.asyncio
async def test_reclaim_once_processes_reclaimed_deliveries_and_acks_after_commit() -> None:
    transaction = RecordingTransaction()
    queue = RecordingQueue(make_delivery())
    failed_processor = DurableEventProcessor(
        make_worker(transaction=RecordingTransaction(fail=True)),
        queue,
        consumer="worker-a",
    )
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        await failed_processor.process_once()
    processor = DurableEventProcessor(
        make_worker(transaction=transaction),
        queue,
        consumer="worker-b",
        reclaim_idle_ms=0,
    )

    processed = await processor.reclaim_once()

    assert processed == 1
    assert transaction.committed is True
    assert queue.pending_delivery_ids() == []
    assert queue.ack_order == ["delivery-1"]


@pytest.mark.asyncio
async def test_stale_transaction_result_remains_pending_until_reprocessed() -> None:
    transaction = RecordingTransaction(
        result=CommitResult(status="stale", reason="conversation_version_mismatch")
    )
    queue = RecordingQueue(make_delivery())
    processor = DurableEventProcessor(
        make_worker(transaction=transaction),
        queue,
        consumer="worker-a",
    )

    result = await processor.process_once()

    assert result is not None
    assert result.status == "stale_or_duplicate"
    assert result.reason == "conversation_version_conflict"
    assert transaction.commands == []
    assert queue.ack_order == []
    assert queue.pending_delivery_ids() == ["delivery-1"]

    transaction.result = None
    recovered = await processor.reclaim_once()

    assert recovered == 1
    assert queue.ack_order == ["delivery-1"]
    assert queue.pending_delivery_ids() == []


@pytest.mark.asyncio
async def test_two_worker_version_race_keeps_loser_delivery_recoverable() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox)
    ready = asyncio.Event()
    entered = 0

    class RaceTransaction:
        async def commit_event(self, *args, **kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            return await transaction.commit_event(*args, **kwargs)

        async def mark_handoff_complete(self, request_id: str) -> None:
            await transaction.mark_handoff_complete(request_id)

    def worker() -> ConversationWorker:
        return ConversationWorker(
            conversations=conversations,
            outbox=outbox,
            tools=FakeBusinessTools(),
            identity=FakeIdentity(),
            knowledge=FakeKnowledge(),
            model=FakeModel(),
            templates=FakeTemplateCatalog(),
            handoff=FakeHandoff(),
            transaction=RaceTransaction(),
            clock=lambda: NOW,
        )

    first_event = make_event(event_key="race-1")
    second_event = replace(
        make_event(event_key="race-2"),
        occurred_at=NOW + timedelta(seconds=1),
        received_at=NOW + timedelta(seconds=1),
    )
    first_queue = RecordingQueue(make_delivery(delivery_id="delivery-1", event=first_event))
    second_queue = RecordingQueue(make_delivery(delivery_id="delivery-2", event=second_event))
    first_processor = DurableEventProcessor(worker(), first_queue, consumer="worker-a")
    second_processor = DurableEventProcessor(worker(), second_queue, consumer="worker-b")

    first_result, second_result = await asyncio.gather(
        first_processor.process_once(), second_processor.process_once()
    )

    results = [first_result, second_result]
    assert sum(result is not None and result.reason == "conversation_version_conflict" for result in results) == 1
    pending = [queue for queue in (first_queue, second_queue) if queue.pending_delivery_ids()]
    assert len(pending) == 1

    recovering = DurableEventProcessor(worker(), pending[0], consumer="worker-c", reclaim_idle_ms=0)
    assert await recovering.reclaim_once() == 1
    assert pending[0].pending_delivery_ids() == []


@pytest.mark.asyncio
async def test_durable_handoff_commits_before_calling_handoff_provider() -> None:
    transaction = RecordingTransaction()
    queue = RecordingQueue(make_delivery(event=make_event(text="tell me a joke")))
    transaction.log = queue.log
    handoff = RecordingHandoff(queue.log)
    processor = DurableEventProcessor(
        make_worker(transaction=transaction, handoff=handoff),
        queue,
        consumer="worker-a",
    )

    result = await processor.process_once()

    assert result is not None
    assert result.status == "handed_off"
    assert result.handoff_created is True
    assert queue.log == [
        "commit:event-1",
        f"handoff:handoff:{result.trace_id}",
        "ack:delivery-1",
    ]


@pytest.mark.asyncio
async def test_durable_handoff_commit_failure_does_not_call_handoff_provider() -> None:
    transaction = RecordingTransaction(fail=True)
    handoff = RecordingHandoff()
    worker = make_worker(transaction=transaction, handoff=handoff)

    with pytest.raises(RuntimeError, match="injected transaction failure"):
        await worker.process(make_event(text="tell me a joke"))

    assert handoff.calls == []


@pytest.mark.asyncio
async def test_durable_handoff_provider_failure_replays_pending_intent() -> None:
    transaction = RecordingTransaction()
    handoff = RecordingHandoff(fail=True)
    queue = RecordingQueue(make_delivery(event=make_event(text="tell me a joke")))
    processor = DurableEventProcessor(
        make_worker(transaction=transaction, handoff=handoff),
        queue,
        consumer="worker-a",
        reclaim_idle_ms=0,
    )

    result = await processor.process_once()

    committed = transaction.conversations["conversation-1"]
    assert result is not None
    assert result.status == "handoff_failed"
    assert result.handoff_created is False
    assert committed.automation_state is AutomationState.HUMAN_ACTIVE
    assert queue.ack_order == []
    assert len(transaction.pending_handoffs) == 1

    first_request = handoff.calls[0]
    handoff.fail = False
    assert await processor.reclaim_once() == 1

    assert [request.request_id for request in handoff.calls] == [
        first_request.request_id,
        first_request.request_id,
    ]
    assert transaction.pending_handoffs == {}
    assert transaction.completed_handoffs == {first_request.request_id}
    assert queue.ack_order == ["delivery-1"]


@pytest.mark.asyncio
async def test_in_memory_transaction_duplicate_dedup_commits_conversation_and_event() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox)
    original = Conversation(
        conversation_id="conversation-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
    )
    updated = replace(original, version=1, customer_last_inbound_at=NOW)
    await conversations.save(original)
    await outbox.enqueue(make_command(dedup_key="reply:event-1"))

    result = await transaction.commit_event(
        event=make_event(),
        expected_version=0,
        conversation=updated,
        command=make_command(dedup_key="reply:event-1"),
    )

    assert result == CommitResult(status="duplicate", reason="dedup_key_exists")
    assert await conversations.get("conversation-1") == updated

    replay = await transaction.commit_event(
        event=make_event(),
        expected_version=1,
        conversation=updated,
        command=None,
    )
    assert replay == CommitResult(status="duplicate", reason="event_already_processed")


@pytest.mark.asyncio
async def test_human_active_durable_commit_never_creates_outbox_command() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox)
    await conversations.save(
        Conversation(
            conversation_id="conversation-1",
            contact_id="contact-1",
            phone_number_id="phone-1",
            automation_state=AutomationState.HUMAN_ACTIVE,
        )
    )
    worker = ConversationWorker(
        conversations=conversations,
        outbox=outbox,
        tools=FakeBusinessTools(),
        identity=FakeIdentity(),
        knowledge=FakeKnowledge(),
        model=FakeModel(),
        templates=FakeTemplateCatalog(),
        handoff=FakeHandoff(),
        transaction=transaction,
        clock=lambda: NOW,
    )

    result = await worker.process(make_event(text="Is this in stock?"))

    assert result.status == "human_active"
    assert result.outbox_created is False
    assert await outbox.records() == []


@pytest.mark.asyncio
async def test_transaction_adapter_suppresses_supplied_command_for_human_active() -> None:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox)
    event = make_event(event_key="human-direct")
    conversation = Conversation(
        conversation_id=event.conversation_id,
        contact_id=event.contact_id,
        phone_number_id=event.phone_number_id,
        customer_last_inbound_at=event.occurred_at,
        automation_state=AutomationState.HUMAN_ACTIVE,
        version=1,
    )

    result = await transaction.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=make_command(dedup_key="must-not-send"),
    )

    assert result == CommitResult(status="committed")
    assert await conversations.get(event.conversation_id) == conversation
    assert await outbox.records() == []
