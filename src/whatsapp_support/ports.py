from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from whatsapp_support.domain import (
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
    QueueDelivery,
    ProviderSendResult,
    RetrievalQuery,
    RagAnswerResult,
    RetrievedChunk,
    StructuredModelResult,
    TemplateDefinition,
    ToolRequest,
    ToolResult,
)


class QueueUnavailable(RuntimeError):
    """Raised when an inbound event cannot be published to the queue."""


class ConcurrentUpdateError(Exception):
    """Raised when an optimistic conversation update loses a version race."""


class InboxPort(Protocol):
    async def record_once(self, event: InboundEvent) -> bool: ...

    async def mark_processed(self, event_key: str) -> None: ...


class ConversationPort(Protocol):
    async def get(self, conversation_id: str) -> Conversation | None: ...

    async def save(self, conversation: Conversation) -> None: ...


class QueuePort(Protocol):
    async def publish(self, event: InboundEvent) -> None: ...


class QueueConsumerPort(Protocol):
    async def receive(
        self, *, consumer: str, block_ms: int = 0
    ) -> QueueDelivery | None: ...

    async def reclaim(
        self, *, consumer: str, min_idle_ms: int
    ) -> list[QueueDelivery]: ...

    async def acknowledge(self, delivery_id: str) -> None: ...


class OutboxPort(Protocol):
    async def enqueue(self, command: OutboundCommand) -> None: ...

    async def claim_pending(
        self,
        now: datetime | None = None,
        lease_seconds: int | None = None,
        max_attempts: int | None = None,
    ) -> OutboxRecord | None: ...

    async def reclaim_expired(
        self, now: datetime, max_attempts: int | None = None
    ) -> int: ...

    async def mark_sent(
        self,
        dedup_key: str,
        provider_message_id: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None: ...

    async def mark_retryable(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None: ...

    async def mark_dead_letter(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None: ...


class ConversationOutboxTransactionPort(Protocol):
    async def commit_event(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        command: OutboundCommand | None,
        handoff_request: HandoffRequest | None = None,
    ) -> CommitResult: ...

    async def mark_handoff_complete(self, request_id: str) -> None: ...


class WhatsAppPort(Protocol):
    async def send(self, command: OutboundCommand) -> ProviderSendResult: ...


class BusinessToolPort(Protocol):
    async def execute(self, request: ToolRequest) -> ToolResult: ...


class IdentityPort(Protocol):
    async def verify(
        self, conversation_id: str, proof: IdentityProof
    ) -> IdentityResult: ...


class KnowledgePort(Protocol):
    async def search(self, query: RetrievalQuery) -> list[Evidence]: ...


class EmbeddingPort(Protocol):
    model_name: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class KnowledgeStorePort(Protocol):
    async def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None: ...


class ModelPort(Protocol):
    async def generate(self, request: ModelRequest) -> StructuredModelResult: ...


class TemplateCatalogPort(Protocol):
    async def resolve(self, name: str, locale: str) -> TemplateDefinition | None: ...


class HandoffPort(Protocol):
    async def create(self, request: HandoffRequest) -> HandoffResult: ...


class ConversationMemoryPort(Protocol):
    async def get_recent_turns(
        self, conversation_id: str, *, limit: int
    ) -> list[ConversationTurn]: ...

    async def get_summary(
        self, conversation_id: str
    ) -> ConversationSummary | None: ...

    async def save_summary(
        self, conversation_id: str, summary: ConversationSummary
    ) -> None: ...


class HybridKnowledgePort(Protocol):
    async def retrieve(
        self, query: RetrievalQuery
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class FollowupRoutingPort(Protocol):
    async def resolve_followup_intent(self, event: InboundEvent) -> str | None: ...


class RagAnswerPort(Protocol):
    async def answer(
        self,
        event: InboundEvent,
        locale: str,
        trace_id: str,
    ) -> RagAnswerResult: ...


class SummaryModelPort(Protocol):
    async def summarize(
        self,
        conversation_id: str,
        turns: list[ConversationTurn],
        existing_summary: ConversationSummary | None,
        trace_id: str,
    ) -> ConversationSummary: ...
