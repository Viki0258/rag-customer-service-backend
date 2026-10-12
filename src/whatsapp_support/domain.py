from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


def _require_utc_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be in UTC")
    return value


def _require_optional_utc_aware(value: datetime | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _require_utc_aware(value, field_name)


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class EventType(str, Enum):
    MESSAGE = "message"
    STATUS = "status"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class AutomationState(str, Enum):
    BOT_ACTIVE = "BOT_ACTIVE"
    HUMAN_ACTIVE = "HUMAN_ACTIVE"
    CLOSED = "CLOSED"


class IdentityState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"


class WindowMode(str, Enum):
    FREEFORM = "freeform"
    TEMPLATE = "template"
    BLOCKED = "blocked"


class OutboundMode(str, Enum):
    FREEFORM = "freeform"
    TEMPLATE = "template"


class OutboxState(str, Enum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    RETRYABLE_FAILED = "RETRYABLE_FAILED"
    SENT = "SENT"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True)
class IdentityProof:
    kind: str
    value: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    source: str
    collected_at: datetime
    detail: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_utc_aware(self.collected_at, "collected_at")


@dataclass(frozen=True)
class InboundEvent:
    event_key: str
    phone_number_id: str
    conversation_id: str
    contact_id: str
    recipient_id: str
    provider_message_id: str | None
    event_type: EventType
    occurred_at: datetime
    received_at: datetime
    payload_digest: str
    text: str | None = None
    identity_proof: IdentityProof | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.recipient_id, "recipient_id")
        _require_utc_aware(self.occurred_at, "occurred_at")
        _require_utc_aware(self.received_at, "received_at")


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    contact_id: str
    phone_number_id: str
    customer_last_inbound_at: datetime | None = None
    window_expires_at: datetime | None = None
    automation_state: AutomationState = AutomationState.BOT_ACTIVE
    identity_state: IdentityState = IdentityState.UNVERIFIED
    identity_subject_id: str | None = None
    identity_scopes: frozenset[str] = frozenset()
    identity_expires_at: datetime | None = None
    version: int = 0

    def __post_init__(self) -> None:
        _require_optional_utc_aware(
            self.customer_last_inbound_at, "customer_last_inbound_at"
        )
        _require_optional_utc_aware(self.window_expires_at, "window_expires_at")
        _require_optional_utc_aware(self.identity_expires_at, "identity_expires_at")

    def apply_inbound(self, event: InboundEvent) -> tuple["Conversation", bool]:
        if event.event_type is not EventType.MESSAGE:
            return self, False

        if (
            self.customer_last_inbound_at is not None
            and event.occurred_at <= self.customer_last_inbound_at
        ):
            return self, False

        updated = replace(
            self,
            customer_last_inbound_at=event.occurred_at,
            window_expires_at=event.occurred_at + timedelta(hours=24),
            version=self.version + 1,
        )
        return updated, True


@dataclass(frozen=True)
class OutboundCommand:
    dedup_key: str
    conversation_id: str
    recipient_id: str
    mode: OutboundMode
    text: str | None
    template_name: str | None
    template_locale: str | None
    template_variables: dict[str, str]
    policy_decision: str
    trace_id: str
    retrieved_chunk_ids: tuple[str, ...] = ()
    supported_chunk_ids: tuple[str, ...] = ()
    prompt_version: str | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.recipient_id, "recipient_id")
        object.__setattr__(self, "retrieved_chunk_ids", tuple(self.retrieved_chunk_ids))
        object.__setattr__(self, "supported_chunk_ids", tuple(self.supported_chunk_ids))


@dataclass(frozen=True)
class OutboxRecord:
    command: OutboundCommand
    state: OutboxState = OutboxState.PENDING
    attempts: int = 0
    provider_message_id: str | None = None
    claimed_at: datetime | None = None
    lease_until: datetime | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        _require_optional_utc_aware(self.claimed_at, "claimed_at")
        _require_optional_utc_aware(self.lease_until, "lease_until")


@dataclass(frozen=True)
class QueueDelivery:
    delivery_id: str
    event: InboundEvent


@dataclass(frozen=True)
class CommitResult:
    status: str
    reason: str | None = None
    handoff_request: HandoffRequest | None = None


@dataclass(frozen=True)
class TemplateDefinition:
    name: str
    locale: str
    category: str
    body_template: str
    variable_names: tuple[str, ...] = ()
    header_template: str | None = None
    footer_text: str | None = None


@dataclass(frozen=True)
class ProviderSendResult:
    provider_message_id: str
    accepted: bool
    sent_at: datetime
    status: str
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _require_utc_aware(self.sent_at, "sent_at")


@dataclass(frozen=True)
class ToolRequest:
    request_id: str
    conversation_id: str
    tool_name: str
    arguments: dict[str, Any]
    trace_id: str
    requested_at: datetime

    def __post_init__(self) -> None:
        _require_utc_aware(self.requested_at, "requested_at")


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    tool_name: str
    success: bool
    output_text: str
    output_data: dict[str, Any]
    completed_at: datetime
    error_code: str | None = None

    def __post_init__(self) -> None:
        _require_utc_aware(self.completed_at, "completed_at")


@dataclass(frozen=True)
class RetrievalQuery:
    query_text: str
    conversation_id: str
    top_k: int
    locale: str | None = None
    market: str | None = None
    trace_id: str | None = None
    query_vector: tuple[float, ...] | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    # None means unrestricted; an empty tuple means no permitted documents.
    allowed_document_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StructuredModelResult:
    trace_id: str
    model_name: str
    prompt_version: str
    content: dict[str, Any]
    raw_text: str
    created_at: datetime
    tokens_in: int = 0
    tokens_out: int = 0

    def __post_init__(self) -> None:
        _require_utc_aware(self.created_at, "created_at")


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    conversation_id: str
    prompt_name: str
    prompt_version: str
    inputs: dict[str, Any]
    trace_id: str
    requested_at: datetime

    def __post_init__(self) -> None:
        _require_utc_aware(self.requested_at, "requested_at")


@dataclass(frozen=True)
class IdentityResult:
    verified: bool
    scopes: frozenset[str]
    subject_id: str
    status: str
    evidence: tuple[Evidence, ...] = ()
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_optional_utc_aware(self.expires_at, "expires_at")

    def allows(self, scope: str) -> bool:
        if scope not in self.scopes:
            return False
        if self.verified:
            return True
        return scope.startswith("public:")


@dataclass(frozen=True)
class HandoffRequest:
    request_id: str
    conversation_id: str
    reason: str
    requested_by: str
    requested_at: datetime
    note: str | None = None

    def __post_init__(self) -> None:
        _require_utc_aware(self.requested_at, "requested_at")


@dataclass(frozen=True)
class HandoffResult:
    request_id: str
    conversation_id: str
    accepted: bool
    queue_name: str
    assigned_agent_id: str | None
    accepted_at: datetime
    note: str | None = None

    def __post_init__(self) -> None:
        _require_utc_aware(self.accepted_at, "accepted_at")


@dataclass(frozen=True)
class Document:
    document_id: str
    source_uri: str
    title: str
    text: str
    content_type: str
    checksum: str
    created_at: datetime
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_utc_aware(self.created_at, "created_at")


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    embedding_input: str
    checksum: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    locale: str
    text: str
    vector_rank: int
    keyword_rank: int
    fused_score: float
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.chunk_id, "chunk_id")
        _require_non_empty_string(self.document_id, "document_id")
        _require_non_empty_string(self.locale, "locale")
        _require_non_empty_string(self.text, "text")


@dataclass(frozen=True)
class RagAnswerResult:
    status: str
    reply: str | None
    locale: str
    trace_id: str
    retrieved_chunk_ids: tuple[str, ...] | list[str]
    supported_chunk_ids: tuple[str, ...] | list[str]
    prompt_version: str
    model_name: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"reply", "clarify", "handoff"}:
            raise ValueError("status must be reply, clarify or handoff")
        object.__setattr__(self, "retrieved_chunk_ids", tuple(self.retrieved_chunk_ids))
        object.__setattr__(self, "supported_chunk_ids", tuple(self.supported_chunk_ids))
        _require_non_empty_string(self.locale, "locale")
        _require_non_empty_string(self.trace_id, "trace_id")
        _require_non_empty_string(self.prompt_version, "prompt_version")
        _require_non_empty_string(self.model_name, "model_name")
        if self.status in {"reply", "clarify"} and (not isinstance(self.reply, str) or not self.reply.strip()):
            raise ValueError("reply must be a non-empty string for reply status")
        if self.status == "handoff" and self.reply is not None:
            raise ValueError("reply must be None for handoff status")


@dataclass(frozen=True)
class ConversationTurn:
    customer_text: str
    assistant_text: str
    event_key: str
    dedup_key: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.customer_text, "customer_text")
        _require_non_empty_string(self.assistant_text, "assistant_text")
        _require_non_empty_string(self.event_key, "event_key")
        _require_non_empty_string(self.dedup_key, "dedup_key")


@dataclass(frozen=True)
class ConversationSummary:
    version: int
    covered_until: datetime
    data: dict[str, Any]
    covered_message_id: int | None = None

    def __post_init__(self) -> None:
        _require_utc_aware(self.covered_until, "covered_until")
        required_keys = {
            "customer_goal",
            "confirmed_entities",
            "customer_preferences",
            "unresolved_questions",
            "decisions",
            "handoff_state",
        }
        if not isinstance(self.data, dict) or not required_keys.issubset(self.data):
            raise ValueError("data must contain the required summary fields")
        if self.covered_message_id is not None and (
            isinstance(self.covered_message_id, bool)
            or not isinstance(self.covered_message_id, int)
            or self.covered_message_id < 1
        ):
            raise ValueError("covered_message_id must be a positive integer")
        object.__setattr__(self, "data", dict(self.data))


def window_mode(conversation: Conversation, now: datetime) -> WindowMode:
    _require_utc_aware(now, "now")
    if conversation.automation_state is AutomationState.HUMAN_ACTIVE:
        return WindowMode.BLOCKED
    if (
        conversation.window_expires_at is not None
        and now < conversation.window_expires_at
    ):
        return WindowMode.FREEFORM
    return WindowMode.TEMPLATE
