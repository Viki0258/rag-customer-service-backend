from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from whatsapp_support.domain import EventType, InboundEvent
from whatsapp_support.gateway import verify_signature
from whatsapp_support.ports import InboxPort, QueuePort


@dataclass(frozen=True)
class MetaEventOutcome:
    event_key: str
    duplicate: bool
    queued: bool
    error_code: str | None = None


@dataclass(frozen=True)
class MetaWebhookResult:
    status_code: int
    accepted: bool
    retryable: bool
    event_results: tuple[MetaEventOutcome, ...]
    unqueued_event_keys: tuple[str, ...]
    error_code: str | None = None


def _require_object(value: object, field_name: str = "payload") -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def _objects(value: object, field_name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return [_require_object(item, field_name) for item in value]


def _required_string(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, object], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _provider_timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be a Unix timestamp")
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a Unix timestamp") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"{field_name} must be a finite Unix timestamp")
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid Unix timestamp") from exc


def _received_at(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("received_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("received_at must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_digest(payload: dict[str, object]) -> str:
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("event payload is not canonicalizable JSON") from exc
    return hashlib.sha256(canonical).hexdigest()


def _event_key(
    phone_number_id: str,
    event_type: EventType,
    provider_message_id: str | None,
    payload_digest: str,
) -> str:
    identifier = provider_message_id or payload_digest
    return f"{phone_number_id}:{event_type.value}:{identifier}"


def _message_event(
    message: dict[str, object],
    phone_number_id: str,
    contacts: set[str],
    received_at: datetime,
) -> InboundEvent:
    wa_id = _required_string(message, "from")
    if contacts and wa_id not in contacts:
        raise ValueError("message sender is absent from contacts")
    provider_message_id = _optional_string(message, "id")
    event_type_value = _required_string(message, "type")
    occurred_at = _provider_timestamp(message.get("timestamp"), "timestamp")
    event_type = EventType.MESSAGE if event_type_value == "text" else EventType.UNSUPPORTED
    text: str | None = None
    if event_type is EventType.MESSAGE:
        text_payload = _require_object(message.get("text"), "text")
        text = _required_string(text_payload, "body")

    payload_digest = _canonical_digest(message)
    return InboundEvent(
        event_key=_event_key(
            phone_number_id, EventType.MESSAGE, provider_message_id, payload_digest
        ),
        phone_number_id=phone_number_id,
        conversation_id=f"{phone_number_id}:{wa_id}",
        contact_id=wa_id,
        recipient_id=wa_id,
        provider_message_id=provider_message_id,
        event_type=event_type,
        occurred_at=occurred_at,
        received_at=received_at,
        payload_digest=payload_digest,
        text=text,
    )


def _status_event(
    status: dict[str, object],
    phone_number_id: str,
    received_at: datetime,
) -> InboundEvent:
    wa_id = _required_string(status, "recipient_id")
    provider_message_id = _optional_string(status, "id")
    occurred_at = _provider_timestamp(status.get("timestamp"), "timestamp")
    payload_digest = _canonical_digest(status)
    return InboundEvent(
        event_key=_event_key(
            phone_number_id, EventType.STATUS, provider_message_id, payload_digest
        ),
        phone_number_id=phone_number_id,
        conversation_id=f"{phone_number_id}:{wa_id}",
        contact_id=wa_id,
        recipient_id=wa_id,
        provider_message_id=provider_message_id,
        event_type=EventType.STATUS,
        occurred_at=occurred_at,
        received_at=received_at,
        payload_digest=payload_digest,
        text=None,
    )


def parse_meta_events(payload: object, received_at: datetime) -> list[InboundEvent]:
    """Normalize Meta's nested webhook envelope into domain events."""

    root = _require_object(payload)
    normalized_received_at = _received_at(received_at)
    events: list[InboundEvent] = []
    for entry in _objects(root.get("entry"), "entry"):
        for change in _objects(entry.get("changes"), "changes"):
            value = _require_object(change.get("value"), "value")
            metadata = _require_object(value.get("metadata"), "metadata")
            phone_number_id = _required_string(metadata, "phone_number_id")
            contacts = {
                _required_string(contact, "wa_id")
                for contact in _objects(value.get("contacts"), "contacts")
            }
            events.extend(
                _message_event(message, phone_number_id, contacts, normalized_received_at)
                for message in _objects(value.get("messages"), "messages")
            )
            events.extend(
                _status_event(status, phone_number_id, normalized_received_at)
                for status in _objects(value.get("statuses"), "statuses")
            )
    if not events:
        raise ValueError("Meta payload contains no parseable events")
    return events


class MetaWebhookAdapter:
    def __init__(
        self,
        inbox: InboxPort,
        queue: QueuePort,
        app_secret: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._inbox = inbox
        self._queue = queue
        self._app_secret = app_secret
        self._clock = clock or (lambda: datetime.now(UTC))

    async def receive(
        self,
        raw_body: bytes,
        signature_header: str | None,
    ) -> MetaWebhookResult:
        if not verify_signature(raw_body, signature_header, self._app_secret):
            return MetaWebhookResult(
                status_code=401,
                accepted=False,
                retryable=False,
                event_results=(),
                unqueued_event_keys=(),
                error_code="invalid_signature",
            )

        try:
            payload = json.loads(raw_body.decode("utf-8"))
            events = parse_meta_events(payload, self._clock())
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return MetaWebhookResult(
                status_code=400,
                accepted=False,
                retryable=False,
                event_results=(),
                unqueued_event_keys=(),
                error_code="invalid_payload",
            )

        outcomes: list[MetaEventOutcome] = []
        unqueued_keys: list[str] = []
        for event in events:
            try:
                recorded = await self._inbox.record_once(event)
            except Exception:
                return MetaWebhookResult(
                    status_code=500,
                    accepted=False,
                    retryable=False,
                    event_results=tuple(outcomes),
                    unqueued_event_keys=tuple(unqueued_keys),
                    error_code="persistence_failed",
                )

            try:
                pending = event if recorded else await self._pending_event(event.event_key)
            except Exception:
                return MetaWebhookResult(
                    status_code=500,
                    accepted=False,
                    retryable=False,
                    event_results=tuple(outcomes),
                    unqueued_event_keys=tuple(unqueued_keys),
                    error_code="persistence_failed",
                )
            if pending is None:
                outcomes.append(MetaEventOutcome(event.event_key, True, False))
                continue

            try:
                await self._queue.publish(pending)
            except Exception:
                if event.event_key not in unqueued_keys:
                    unqueued_keys.append(event.event_key)
                outcomes.append(
                    MetaEventOutcome(
                        event.event_key,
                        not recorded,
                        False,
                        "queue_unavailable",
                    )
                )
                continue

            try:
                await self._mark_queued(event.event_key)
            except Exception:
                if event.event_key not in unqueued_keys:
                    unqueued_keys.append(event.event_key)
                outcomes.append(
                    MetaEventOutcome(
                        event.event_key,
                        not recorded,
                        False,
                        "persistence_failed",
                    )
                )
                return MetaWebhookResult(
                    status_code=500,
                    accepted=False,
                    retryable=False,
                    event_results=tuple(outcomes),
                    unqueued_event_keys=tuple(unqueued_keys),
                    error_code="persistence_failed",
                )

            outcomes.append(MetaEventOutcome(event.event_key, not recorded, True))

        if unqueued_keys:
            return MetaWebhookResult(
                status_code=202,
                accepted=True,
                retryable=True,
                event_results=tuple(outcomes),
                unqueued_event_keys=tuple(unqueued_keys),
            )
        return MetaWebhookResult(
            status_code=200,
            accepted=True,
            retryable=False,
            event_results=tuple(outcomes),
            unqueued_event_keys=(),
        )

    async def _pending_event(self, event_key: str) -> InboundEvent | None:
        list_unqueued = getattr(self._inbox, "list_unqueued", None)
        if list_unqueued is None:
            return None
        for event in await list_unqueued():
            if event.event_key == event_key:
                return event
        return None

    async def _mark_queued(self, event_key: str) -> None:
        marker = getattr(self._inbox, "mark_queued", None)
        if marker is not None:
            await marker(event_key)
