from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from whatsapp_support.domain import (
    ConversationSummary,
    ConversationTurn,
    InboundEvent,
    OutboundCommand,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationContext:
    summary: ConversationSummary | None
    recent_turns: tuple[ConversationTurn, ...]
    handoff_state: str
    latest_customer_message: str | None = None


class RequestMemorySnapshot:
    """One conversation's read-only history snapshot for a single request.

    This is not a credential or business fact store. Writes stay in the worker's
    existing conversation/outbox transaction.
    """
    def __init__(self, conversation_id, turns, summary):
        self.conversation_id = conversation_id
        self.turns = tuple(turns)
        self.summary = summary

    @classmethod
    async def load(cls, memory, conversation_id, limit=6):
        turns = await memory.get_recent_turns(conversation_id, limit=limit)
        try:
            summary = await memory.get_summary(conversation_id)
        except Exception:
            summary = None
        return cls(conversation_id, turns, summary)

    async def get_recent_turns(self, conversation_id, limit=6):
        if conversation_id != self.conversation_id:
            raise ValueError('Context belongs to another conversation')
        return self.turns[-limit:]

    async def get_summary(self, conversation_id):
        if conversation_id != self.conversation_id:
            raise ValueError('Context belongs to another conversation')
        return self.summary


@dataclass(slots=True)
class RollingSummaryPolicy:
    max_recent_turns: int = 6
    summarize_at_turns: int = 8
    summarize_fn: Callable[
        [str, list[ConversationTurn], ConversationSummary | None, str],
        ConversationSummary,
    ] | None = None

    def maybe_summarize(
        self,
        conversation_id: str,
        complete_turns: list[ConversationTurn],
        existing_summary: ConversationSummary | None,
        trace_id: str,
    ) -> ConversationSummary | None:
        if len(complete_turns) < self.summarize_at_turns:
            return existing_summary
        if self.summarize_fn is None:
            return existing_summary

        keep = min(self.max_recent_turns, len(complete_turns))
        turns_to_summarize = complete_turns[: len(complete_turns) - keep]
        if not turns_to_summarize:
            return existing_summary

        try:
            summary = self.summarize_fn(
                conversation_id,
                [self._sanitize_turn(turn) for turn in turns_to_summarize],
                existing_summary,
                trace_id,
            )
            return _sanitize_summary(summary)
        except Exception:
            logger.warning(
                "conversation summary failed",
                extra={
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                },
            )
            return existing_summary

    @staticmethod
    def _sanitize_turn(turn: ConversationTurn) -> ConversationTurn:
        return ConversationTurn(
            customer_text=_sanitize_text(turn.customer_text),
            assistant_text=_sanitize_text(turn.assistant_text),
            event_key=turn.event_key,
            dedup_key=turn.dedup_key,
        )


def conservative_structured_summary(
    conversation_id: str,
    turns: list[ConversationTurn],
    existing_summary: ConversationSummary | None,
    trace_id: str,
) -> ConversationSummary:
    del conversation_id, trace_id
    prior_handoff = (
        str(existing_summary.data.get("handoff_state", "none"))
        if existing_summary is not None
        else "none"
    )
    customer_goal = _sanitize_text(turns[-1].customer_text)[:300] if turns else ""
    # Bounded extractive references, not model-generated facts. Keep speakers
    # separate so a past recommendation never becomes a customer preference.
    references = []
    prior = existing_summary.data.get('dialogue_references', []) if existing_summary else []
    for row in [*prior, *({'customer': t.customer_text, 'assistant': t.assistant_text} for t in turns)]:
        if not isinstance(row, dict):
            continue
        item = {role: _sanitize_text(str(row.get(role, '')))[:400]
                for role in ('customer', 'assistant')}
        if item in references:
            references.remove(item)
        references.append(item)
    return ConversationSummary(
        version=1 if existing_summary is None else existing_summary.version + 1,
        covered_until=datetime.now(UTC),
        data={
            "customer_goal": _sanitize_text(customer_goal),
            "confirmed_entities": [],
            "customer_preferences": [],
            "unresolved_questions": [],
            "decisions": [],
            "handoff_state": prior_handoff,
            "dialogue_references": references[-8:],
        },
    )


@dataclass(slots=True)
class _ConversationState:
    customer_records: dict[str, tuple[datetime, str]] = field(default_factory=dict)
    assistant_records: dict[str, tuple[datetime, str]] = field(default_factory=dict)
    message_log: list[_MessageRecord] = field(default_factory=list)
    next_sequence: int = 0
    summary: ConversationSummary | None = None
    handoff_state: str = "none"


@dataclass(slots=True)
class _MessageRecord:
    sequence: int
    kind: str
    key: str
    text: str
    occurred_at: datetime


class InMemoryConversationMemory:
    def __init__(self, *, summary_policy: RollingSummaryPolicy | None = None) -> None:
        self._summary_policy = summary_policy or RollingSummaryPolicy()
        self._states: dict[str, _ConversationState] = {}

    def append_customer(self, event: InboundEvent) -> None:
        state = self._state(event.conversation_id)
        if event.event_key in state.customer_records:
            return
        text = event.text if isinstance(event.text, str) else ""
        state.customer_records[event.event_key] = (
            _event_time(event),
            text,
        )
        state.message_log.append(
            _MessageRecord(
                sequence=state.next_sequence,
                kind="customer",
                key=event.event_key,
                text=text,
                occurred_at=_event_time(event),
            )
        )
        state.next_sequence += 1

    def append_assistant(self, command: OutboundCommand) -> None:
        state = self._state(command.conversation_id)
        if command.dedup_key in state.assistant_records:
            return
        text = _assistant_text(command)
        state.assistant_records[command.dedup_key] = (
            datetime.now(UTC),
            text,
        )
        state.message_log.append(
            _MessageRecord(
                sequence=state.next_sequence,
                kind="assistant",
                key=command.dedup_key,
                text=text,
                occurred_at=datetime.now(UTC),
            )
        )
        state.next_sequence += 1
        state.summary = self._summary_policy.maybe_summarize(
            command.conversation_id,
            list(self._complete_turns(state)),
            state.summary,
            command.trace_id,
        )
        if state.summary is not None:
            state.handoff_state = str(state.summary.data.get("handoff_state", "none"))

    def save_summary(self, conversation_id: str, summary: ConversationSummary) -> None:
        self._state(conversation_id).summary = summary

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        return self._state(conversation_id).summary

    def context(self, conversation_id: str) -> ConversationContext:
        state = self._state(conversation_id)
        complete_turns = tuple(self._complete_turns(state)[-self._summary_policy.max_recent_turns :])
        latest_customer_message = _latest_customer_message(state)
        summary = state.summary
        if summary is not None:
            state.handoff_state = str(summary.data.get("handoff_state", "none"))
        return ConversationContext(
            summary=summary,
            recent_turns=complete_turns,
            handoff_state=state.handoff_state,
            latest_customer_message=latest_customer_message,
        )

    def _state(self, conversation_id: str) -> _ConversationState:
        state = self._states.get(conversation_id)
        if state is None:
            state = _ConversationState()
            self._states[conversation_id] = state
        return state

    @staticmethod
    def _complete_turns(state: _ConversationState) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        pending_customer: _MessageRecord | None = None
        for message in sorted(state.message_log, key=lambda item: item.sequence):
            if message.kind == "customer":
                pending_customer = message
                continue
            if message.kind != "assistant" or pending_customer is None:
                continue
            turns.append(
                ConversationTurn(
                    customer_text=pending_customer.text,
                    assistant_text=message.text,
                    event_key=pending_customer.key,
                    dedup_key=message.key,
                )
            )
            pending_customer = None
        return turns


class AsyncInMemoryConversationMemory:
    """Async port facade over the deterministic in-memory memory store."""

    def __init__(self, store: InMemoryConversationMemory) -> None:
        self.store = store

    async def get_recent_turns(
        self, conversation_id: str, *, limit: int
    ) -> list[ConversationTurn]:
        if limit <= 0:
            return []
        return list(self.store.context(conversation_id).recent_turns[-limit:])

    async def get_summary(
        self, conversation_id: str
    ) -> ConversationSummary | None:
        return self.store.context(conversation_id).summary

    async def save_summary(
        self, conversation_id: str, summary: ConversationSummary
    ) -> None:
        self.store.save_summary(conversation_id, summary)


def _assistant_text(command: OutboundCommand) -> str:
    if isinstance(command.text, str) and command.text.strip():
        return command.text.strip()
    reply = command.template_variables.get("reply")
    if isinstance(reply, str):
        return reply.strip()
    return ""


def _event_time(event: InboundEvent) -> datetime:
    return event.occurred_at.astimezone(UTC)


def _clean_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return _sanitize_text(text)


def _sanitize_text(text: str) -> str:
    lowered = text.lower()
    if "verification code" in lowered or "验证码" in text or "动态码" in text:
        return "[redacted]"
    if (
        "access token" in lowered
        or "token" in lowered
        or "credential" in lowered
        or "password" in lowered
        or "api key" in lowered
        or "api_key" in lowered
        or "apikey" in lowered
        or "bearer" in lowered
        or "secret" in lowered
        or "访问令牌" in text
        or "凭证" in text
        or "密码" in text
        or "密钥" in text
    ):
        return "[redacted]"
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return "[redacted]"
    return text


def _sanitize_summary(summary: ConversationSummary) -> ConversationSummary:
    return ConversationSummary(
        version=summary.version,
        covered_until=summary.covered_until,
        covered_message_id=summary.covered_message_id,
        data=_sanitize_summary_mapping(summary.data),
    )


def _sanitize_summary_mapping(mapping: dict[Any, Any]) -> dict[Any, Any]:
    return {
        key: _sanitize_summary_value(value)
        for key, value in mapping.items()
        if not isinstance(key, str) or _sanitize_text(key) == key
    }


def _sanitize_summary_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_summary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_summary_value(item) for item in value)
    if isinstance(value, dict):
        return _sanitize_summary_mapping(value)
    return value


_SECRET_PATTERNS = (
    re.compile(r"(?<!\d)\d{6}(?!\d)"),
    re.compile(r"\b(?:[A-Za-z0-9_-]{8,}\.){2}[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
        r"AIza[A-Za-z0-9_-]{20,})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(
        r"\b(?=[A-Za-z0-9_-]{32,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])"
        r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+\b"
    ),
)


def _latest_customer_message(state: _ConversationState) -> str | None:
    if not state.message_log:
        return None
    for message in reversed(state.message_log):
        if message.kind == "customer":
            return message.text
    return None
