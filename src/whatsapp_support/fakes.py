from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Generic, TypeVar

from whatsapp_support.domain import (
    AutomationState,
    CommitResult,
    Conversation,
    ConversationSummary,
    ConversationTurn,
    Evidence,
    HandoffRequest,
    HandoffResult,
    IdentityProof,
    IdentityResult,
    InboundEvent,
    KnowledgeChunk,
    ModelRequest,
    OutboundCommand,
    OutboxRecord,
    OutboxState,
    ProviderSendResult,
    QueueDelivery,
    RagAnswerResult,
    RetrievalQuery,
    RetrievedChunk,
    StructuredModelResult,
    TemplateDefinition,
    ToolRequest,
    ToolResult,
)
from whatsapp_support.memory import ConversationContext, InMemoryConversationMemory, RollingSummaryPolicy
from whatsapp_support.ports import ConcurrentUpdateError, QueueUnavailable


class InMemoryInboxStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._events: OrderedDict[str, InboundEvent] = OrderedDict()
        self._queued: set[str] = set()
        self._processed: set[str] = set()

    async def record_once(self, event: InboundEvent) -> bool:
        async with self._lock:
            if event.event_key in self._events:
                return False
            self._events[event.event_key] = event
            self._queued.discard(event.event_key)
            return True

    async def list_unqueued(self) -> list[InboundEvent]:
        async with self._lock:
            return [
                event
                for event_key, event in self._events.items()
                if event_key not in self._queued and event_key not in self._processed
            ]

    async def mark_queued(self, event_key: str) -> None:
        async with self._lock:
            if event_key in self._events:
                self._queued.add(event_key)

    async def mark_processed(self, event_key: str) -> None:
        async with self._lock:
            self._queued.discard(event_key)
            if event_key in self._events:
                self._processed.add(event_key)

    async def get(self, event_key: str) -> InboundEvent | None:
        async with self._lock:
            return self._events.get(event_key)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._records: dict[str, Conversation] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, conversation_id: str) -> Conversation | None:
        lock = await self._get_lock(conversation_id)
        async with lock:
            return self._records.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        lock = await self._get_lock(conversation.conversation_id)
        async with lock:
            current = self._records.get(conversation.conversation_id)
            if current is not None and conversation.version < current.version:
                raise ConcurrentUpdateError(
                    f"stale conversation version for {conversation.conversation_id}"
                )
            self._records[conversation.conversation_id] = conversation

    async def update_from_event(
        self, event: InboundEvent
    ) -> tuple[Conversation, bool]:
        lock = await self._get_lock(event.conversation_id)
        async with lock:
            current = self._records.get(event.conversation_id) or Conversation(
                conversation_id=event.conversation_id,
                contact_id=event.contact_id,
                phone_number_id=event.phone_number_id,
            )
            updated, changed = current.apply_inbound(event)
            if changed:
                self._records[event.conversation_id] = updated
                return updated, True
            return current, False

    async def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock

class InMemoryQueue:
    def __init__(self, *, fail_next_publish: bool = False) -> None:
        self.fail_next_publish = fail_next_publish
        self._lock = asyncio.Lock()
        self._deliveries: OrderedDict[str, QueueDelivery] = OrderedDict()
        self._counter = 0

    async def publish(self, event: InboundEvent) -> None:
        async with self._lock:
            if self.fail_next_publish:
                self.fail_next_publish = False
                raise QueueUnavailable("in-memory queue publish failed")
            self._counter += 1
            delivery_id = f"delivery-{self._counter}"
            self._deliveries[delivery_id] = QueueDelivery(delivery_id, event)

    async def acknowledge(self, delivery_id: str) -> None:
        async with self._lock:
            self._deliveries.pop(delivery_id, None)

    async def receive(
        self, *, consumer: str, block_ms: int = 0
    ) -> QueueDelivery | None:
        del consumer, block_ms
        async with self._lock:
            for delivery in self._deliveries.values():
                return delivery
            return None

    async def reclaim(
        self, *, consumer: str, min_idle_ms: int
    ) -> list[QueueDelivery]:
        del consumer, min_idle_ms
        return []

    async def take(self) -> tuple[str, InboundEvent] | None:
        async with self._lock:
            for delivery_id, delivery in self._deliveries.items():
                return delivery_id, delivery.event
            return None

    async def pending_delivery_ids(self) -> list[str]:
        async with self._lock:
            return list(self._deliveries.keys())

class InMemoryOutboxStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: OrderedDict[str, OutboxRecord] = OrderedDict()

    async def enqueue(self, command: OutboundCommand) -> None:
        async with self._lock:
            if command.dedup_key in self._records:
                return
            self._records[command.dedup_key] = OutboxRecord(command=command)

    async def claim_pending(
        self,
        now: datetime | None = None,
        lease_seconds: int | None = None,
        max_attempts: int | None = None,
    ) -> OutboxRecord | None:
        claim_time = _utc_now() if now is None else _ensure_utc(now)
        effective_lease_seconds = _effective_lease_seconds(lease_seconds)
        _validate_max_attempts(max_attempts)
        async with self._lock:
            for dedup_key, record in self._records.items():
                expired_lease = (
                    record.state is OutboxState.SENDING
                    and record.lease_until is not None
                    and record.lease_until <= claim_time
                )
                if (
                    record.state
                    not in {
                        OutboxState.PENDING,
                        OutboxState.RETRYABLE_FAILED,
                    }
                    and not expired_lease
                ):
                    continue
                if max_attempts is not None and record.attempts >= max_attempts:
                    self._records[dedup_key] = replace(
                        record,
                        state=OutboxState.DEAD_LETTER,
                        claimed_at=None,
                        lease_until=None,
                        last_error="max_attempts_exhausted_after_lease_expiry",
                    )
                    continue
                claimed = replace(
                    record,
                    state=OutboxState.SENDING,
                    attempts=record.attempts + 1,
                    claimed_at=claim_time,
                    lease_until=claim_time + timedelta(seconds=effective_lease_seconds),
                    last_error="lease_expired" if expired_lease else record.last_error,
                )
                self._records[dedup_key] = claimed
                return claimed
            return None

    async def reclaim_expired(
        self, now: datetime, max_attempts: int | None = None
    ) -> int:
        reclaim_time = _ensure_utc(now)
        _validate_max_attempts(max_attempts)
        reclaimed = 0
        async with self._lock:
            for dedup_key, record in list(self._records.items()):
                if (
                    record.state is OutboxState.SENDING
                    and record.lease_until is not None
                    and record.lease_until <= reclaim_time
                ):
                    self._records[dedup_key] = replace(
                        record,
                        state=(
                            OutboxState.DEAD_LETTER
                            if max_attempts is not None
                            and record.attempts >= max_attempts
                            else OutboxState.RETRYABLE_FAILED
                        ),
                        claimed_at=None,
                        lease_until=None,
                        last_error=(
                            "max_attempts_exhausted_after_lease_expiry"
                            if max_attempts is not None
                            and record.attempts >= max_attempts
                            else "lease_expired"
                        ),
                    )
                    reclaimed += 1
        return reclaimed

    async def mark_sent(
        self,
        dedup_key: str,
        provider_message_id: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get(dedup_key)
            if record is None or not _matches_claim_fence(
                record, attempts=attempts, claimed_at=claimed_at
            ):
                return
            self._records[dedup_key] = replace(
                record,
                state=OutboxState.SENT,
                provider_message_id=provider_message_id,
                claimed_at=None,
                lease_until=None,
                last_error=None,
            )

    async def mark_retryable(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get(dedup_key)
            if record is None or not _matches_claim_fence(
                record, attempts=attempts, claimed_at=claimed_at
            ):
                return
            self._records[dedup_key] = replace(
                record,
                state=OutboxState.RETRYABLE_FAILED,
                claimed_at=None,
                lease_until=None,
                last_error=reason,
            )

    async def mark_dead_letter(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get(dedup_key)
            if record is None or not _matches_claim_fence(
                record, attempts=attempts, claimed_at=claimed_at
            ):
                return
            self._records[dedup_key] = replace(
                record,
                state=OutboxState.DEAD_LETTER,
                claimed_at=None,
                lease_until=None,
                last_error=reason,
            )

    async def records(self) -> list[OutboxRecord]:
        async with self._lock:
            return list(self._records.values())


class InMemoryConversationOutboxTransaction:
    def __init__(
        self,
        conversations: InMemoryConversationStore,
        outbox: InMemoryOutboxStore,
        memory=None,
    ) -> None:
        self._conversations = conversations
        self._outbox = outbox
        self._memory = memory
        self._lock = asyncio.Lock()
        self._processed_event_keys: set[str] = set()
        self._handoff_intents: dict[str, HandoffRequest] = {}
        self._handoff_event_keys: dict[str, str] = {}

    async def commit_event(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        command: OutboundCommand | None,
        handoff_request: HandoffRequest | None = None,
    ) -> CommitResult:
        async with self._lock:
            if event.event_key in self._processed_event_keys:
                request_id = self._handoff_event_keys.get(event.event_key)
                pending = (
                    None if request_id is None else self._handoff_intents.get(request_id)
                )
                return CommitResult(
                    status="duplicate",
                    reason="event_already_processed",
                    handoff_request=pending,
                )

            current = await self._conversations.get(event.conversation_id)
            current_version = 0 if current is None else current.version
            if current_version != expected_version:
                return CommitResult(
                    status="stale", reason="conversation_version_mismatch"
                )

            if conversation.automation_state is AutomationState.HUMAN_ACTIVE:
                command = None

            duplicate_command = False
            if command is not None:
                records = await self._outbox.records()
                duplicate_command = any(
                    record.command.dedup_key == command.dedup_key
                    for record in records
                )

            await self._conversations.save(conversation)
            self._processed_event_keys.add(event.event_key)
            if self._memory is not None:
                self._memory.append_customer(event)
            if handoff_request is not None:
                self._handoff_intents.setdefault(
                    handoff_request.request_id, handoff_request
                )
                self._handoff_event_keys[event.event_key] = handoff_request.request_id
            if command is not None and not duplicate_command:
                await self._outbox.enqueue(command)
                if self._memory is not None:
                    self._memory.append_assistant(command)
            return CommitResult(
                status="duplicate" if duplicate_command else "committed",
                reason="dedup_key_exists" if duplicate_command else None,
                handoff_request=handoff_request,
            )

    async def mark_handoff_complete(self, request_id: str) -> None:
        async with self._lock:
            self._handoff_intents.pop(request_id, None)


_ResultT = TypeVar("_ResultT")


class _QueueDrivenFake(Generic[_ResultT]):
    def __init__(self) -> None:
        self.errors_queue: list[Exception] = []
        self.results_queue: list[_ResultT] = []

    def _next_result(self, default: _ResultT) -> _ResultT:
        if self.errors_queue:
            raise self.errors_queue.pop(0)
        if self.results_queue:
            return self.results_queue.pop(0)
        return default


class FakeBusinessTools(_QueueDrivenFake[ToolResult]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ToolRequest] = []

    async def execute(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        return self._next_result(
            ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                output_text="ok",
                output_data={},
                completed_at=_DEFAULT_TIME,
            )
        )


class FakeIdentity(_QueueDrivenFake[IdentityResult]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, IdentityProof]] = []

    async def verify(
        self, conversation_id: str, proof: IdentityProof
    ) -> IdentityResult:
        self.calls.append((conversation_id, proof))
        return self._next_result(
            IdentityResult(
                verified=False,
                scopes=frozenset(),
                subject_id=conversation_id,
                status="unverified",
                evidence=(),
            )
        )


class FakeKnowledge(_QueueDrivenFake[list[Evidence]]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RetrievalQuery] = []

    async def search(self, query: RetrievalQuery) -> list[Evidence]:
        self.calls.append(query)
        return self._next_result([])


class FakeHybridKnowledge(_QueueDrivenFake[list[RetrievedChunk]]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.calls.append(query)
        return self._next_result([])


class FakeEmbedding(_QueueDrivenFake[list[list[float]]]):
    def __init__(
        self,
        *,
        model_name: str = "fake-embedding",
        dimension: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.model_name = model_name.strip()
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._next_result([[0.0] * self.dimension for _ in texts])


class FakeKnowledgeStore(_QueueDrivenFake[None]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[KnowledgeChunk]] = []

    async def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self.calls.append(list(chunks))
        self._next_result(None)


class FakeModel(_QueueDrivenFake[StructuredModelResult]):
    def __init__(self, *, model_name: str = "fake-model") -> None:
        super().__init__()
        self.model_name = model_name
        self.calls: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> StructuredModelResult:
        self.calls.append(request)
        return self._next_result(
            StructuredModelResult(
                trace_id=request.trace_id,
                model_name=self.model_name,
                prompt_version=request.prompt_version,
                content={},
                raw_text="{}",
                created_at=_DEFAULT_TIME,
            )
        )


class FakeConversationMemory:
    def __init__(
        self,
        *,
        recent_turns: list[ConversationTurn] | None = None,
        summary: ConversationSummary | None = None,
    ) -> None:
        self.recent_turns = list(recent_turns or [])
        self.summary = summary
        self.customer_turns: list[tuple[str, ConversationTurn]] = []
        self.assistant_turns: list[tuple[str, ConversationTurn]] = []
        self.recent_turn_limits: list[int] = []
        self.summary_calls: list[str] = []
        self.saved_summaries: list[tuple[str, ConversationSummary]] = []

    async def append_customer_turn(
        self, conversation_id: str, turn: ConversationTurn
    ) -> None:
        self.customer_turns.append((conversation_id, turn))

    async def append_assistant_turn(
        self, conversation_id: str, turn: ConversationTurn
    ) -> None:
        self.assistant_turns.append((conversation_id, turn))

    async def get_recent_turns(
        self, conversation_id: str, *, limit: int
    ) -> list[ConversationTurn]:
        del conversation_id
        self.recent_turn_limits.append(limit)
        if limit <= 0:
            return []
        return list(self.recent_turns[-limit:])

    async def get_summary(
        self, conversation_id: str
    ) -> ConversationSummary | None:
        self.summary_calls.append(conversation_id)
        return self.summary

    async def save_summary(
        self, conversation_id: str, summary: ConversationSummary
    ) -> None:
        self.saved_summaries.append((conversation_id, summary))
        self.summary = summary


class FakeTemplateCatalog(_QueueDrivenFake[TemplateDefinition | None]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    async def resolve(self, name: str, locale: str) -> TemplateDefinition | None:
        self.calls.append((name, locale))
        return self._next_result(None)


class FakeHandoff(_QueueDrivenFake[HandoffResult]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[HandoffRequest] = []

    async def create(self, request: HandoffRequest) -> HandoffResult:
        self.calls.append(request)
        return self._next_result(
            HandoffResult(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                accepted=True,
                queue_name="human-handoff",
                assigned_agent_id=None,
                accepted_at=_DEFAULT_TIME,
            )
        )


class FakeWhatsApp(_QueueDrivenFake[ProviderSendResult]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[OutboundCommand] = []

    async def send(self, command: OutboundCommand) -> ProviderSendResult:
        self.calls.append(command)
        return self._next_result(
            ProviderSendResult(
                provider_message_id=f"fake-{command.dedup_key}",
                accepted=True,
                sent_at=_DEFAULT_TIME,
                status="accepted",
            )
        )


_DEFAULT_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _effective_lease_seconds(lease_seconds: int | None) -> int:
    if lease_seconds is None:
        return 60
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return lease_seconds


def _validate_max_attempts(max_attempts: int | None) -> None:
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _matches_claim_fence(
    record: OutboxRecord,
    *,
    attempts: int | None,
    claimed_at: datetime | None,
) -> bool:
    if attempts is None and claimed_at is None:
        return True
    if record.state is not OutboxState.SENDING:
        return False
    if attempts is not None and record.attempts != attempts:
        return False
    if claimed_at is not None and record.claimed_at != _ensure_utc(claimed_at):
        return False
    return True
