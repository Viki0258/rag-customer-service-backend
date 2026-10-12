from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from whatsapp_support.domain import EventType, IdentityProof, InboundEvent
from whatsapp_support.ports import InboxPort, QueuePort, QueueUnavailable


class ChallengeRejected(ValueError):
    """Raised when a webhook subscription challenge is not authorized."""


def verify_signature(
    raw_body: bytes,
    signature_header: str | None,
    app_secret: str,
) -> bool:
    """Verify a provider ``sha256=<hex digest>`` signature."""

    if not isinstance(raw_body, bytes) or not isinstance(signature_header, str):
        return False
    if not isinstance(app_secret, str) or not app_secret:
        return False
    prefix = "sha256="
    if not signature_header.startswith(prefix):
        return False

    supplied_digest = signature_header[len(prefix) :]
    if len(supplied_digest) != hashlib.sha256().digest_size * 2:
        return False
    try:
        bytes.fromhex(supplied_digest)
    except ValueError:
        return False

    expected_digest = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_digest, supplied_digest.lower())


def verify_challenge(
    mode: str,
    token: str,
    challenge: str,
    expected_token: str,
) -> str:
    """Return a valid subscription challenge or reject it."""

    if (
        mode != "subscribe"
        or not isinstance(token, str)
        or not isinstance(expected_token, str)
        or not hmac.compare_digest(token, expected_token)
    ):
        raise ChallengeRejected("webhook challenge rejected")
    return challenge


def _utc_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC")
    return value.astimezone(UTC)


def _parse_utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC string")
    timestamp = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC string") from exc
    return _utc_datetime(parsed, field_name)


def _required_string(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_payload(payload: dict[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload is not canonicalizable JSON") from exc


def normalize_event(
    payload: dict[str, object],
    received_at: datetime,
) -> InboundEvent:
    """Convert the flat fake-provider payload into a domain event."""

    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    phone_number_id = _required_string(payload, "phone_number_id")
    conversation_id = _required_string(payload, "conversation_id")
    contact_id = _required_string(payload, "contact_id")
    recipient_id = _required_string(payload, "recipient_id")
    event_type_value = _required_string(payload, "event_type")
    try:
        event_type = EventType(event_type_value)
    except ValueError as exc:
        raise ValueError("unsupported event_type") from exc

    occurred_at = _parse_utc_timestamp(payload.get("occurred_at"), "occurred_at")
    normalized_received_at = _utc_datetime(received_at, "received_at")

    provider_message_id_value = payload.get("provider_message_id")
    if provider_message_id_value is not None and not isinstance(
        provider_message_id_value, str
    ):
        raise ValueError("provider_message_id must be a string")
    provider_message_id = provider_message_id_value or None

    text = payload.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError("text must be a string")

    identity_proof_value = payload.get("identity_proof")
    identity_proof: IdentityProof | None = None
    if identity_proof_value is not None:
        if not isinstance(identity_proof_value, dict):
            raise ValueError("identity_proof must be an object")
        kind = identity_proof_value.get("kind")
        proof_value = identity_proof_value.get("value")
        if not isinstance(kind, str) or not kind:
            raise ValueError("identity_proof.kind must be a non-empty string")
        if not isinstance(proof_value, str) or not proof_value:
            raise ValueError("identity_proof.value must be a non-empty string")
        identity_proof = IdentityProof(kind=kind, value=proof_value)

    canonical_payload = _canonical_payload(payload)
    payload_digest = hashlib.sha256(canonical_payload).hexdigest()
    if event_type in {EventType.MESSAGE, EventType.STATUS} and provider_message_id:
        event_key = f"{phone_number_id}:{event_type.value}:{provider_message_id}"
    else:
        event_key = f"{phone_number_id}:{event_type.value}:{payload_digest}"

    return InboundEvent(
        event_key=event_key,
        phone_number_id=phone_number_id,
        conversation_id=conversation_id,
        contact_id=contact_id,
        recipient_id=recipient_id,
        provider_message_id=provider_message_id,
        event_type=event_type,
        occurred_at=occurred_at,
        received_at=normalized_received_at,
        payload_digest=payload_digest,
        text=text,
        identity_proof=identity_proof,
    )


@dataclass(frozen=True)
class WebhookResult:
    status_code: int
    accepted: bool
    duplicate: bool
    queued: bool
    event_key: str | None
    error_code: str | None = None


def _decode_payload(raw_body: bytes) -> dict[str, object]:
    if not isinstance(raw_body, bytes):
        raise ValueError("raw body must be bytes")
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("body is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("payload must be a JSON object")
    return decoded


class WebhookGateway:
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
    ) -> WebhookResult:
        if not verify_signature(raw_body, signature_header, self._app_secret):
            return WebhookResult(
                status_code=401,
                accepted=False,
                duplicate=False,
                queued=False,
                event_key=None,
                error_code="invalid_signature",
            )

        try:
            event = normalize_event(_decode_payload(raw_body), self._clock())
        except (TypeError, ValueError):
            return WebhookResult(
                status_code=400,
                accepted=False,
                duplicate=False,
                queued=False,
                event_key=None,
                error_code="invalid_payload",
            )

        try:
            recorded = await self._inbox.record_once(event)
        except Exception:
            return WebhookResult(
                status_code=500,
                accepted=False,
                duplicate=False,
                queued=False,
                event_key=event.event_key,
                error_code="persistence_failed",
            )

        if not recorded:
            retry_result = await self._retry_unqueued(event)
            if retry_result is not None:
                return retry_result
            return WebhookResult(
                status_code=200,
                accepted=True,
                duplicate=True,
                queued=False,
                event_key=event.event_key,
            )

        try:
            await self._queue.publish(event)
        except QueueUnavailable:
            return WebhookResult(
                status_code=202,
                accepted=True,
                duplicate=False,
                queued=False,
                event_key=event.event_key,
                error_code="queue_unavailable",
            )

        await self._mark_queued(event.event_key)
        return WebhookResult(
            status_code=200,
            accepted=True,
            duplicate=False,
            queued=True,
            event_key=event.event_key,
        )

    async def _mark_queued(self, event_key: str) -> None:
        marker = getattr(self._inbox, "mark_queued", None)
        if marker is not None:
            await marker(event_key)

    async def _retry_unqueued(self, event: InboundEvent) -> WebhookResult | None:
        list_unqueued = getattr(self._inbox, "list_unqueued", None)
        if list_unqueued is None:
            return None
        pending = next(
            (
                stored
                for stored in await list_unqueued()
                if stored.event_key == event.event_key
            ),
            None,
        )
        if pending is None:
            return None
        try:
            await self._queue.publish(pending)
        except QueueUnavailable:
            return WebhookResult(
                status_code=202,
                accepted=True,
                duplicate=True,
                queued=False,
                event_key=event.event_key,
                error_code="queue_unavailable",
            )
        await self._mark_queued(event.event_key)
        return WebhookResult(
            status_code=200,
            accepted=True,
            duplicate=True,
            queued=True,
            event_key=event.event_key,
        )


class InboxRelay:
    def __init__(self, inbox: InboxPort, queue: QueuePort) -> None:
        self._inbox = inbox
        self._queue = queue

    async def relay_once(self) -> int:
        list_unqueued = getattr(self._inbox, "list_unqueued", None)
        if list_unqueued is None:
            return 0

        relayed = 0
        for event in await list_unqueued():
            try:
                await self._queue.publish(event)
            except QueueUnavailable:
                continue
            await self._mark_queued(event.event_key)
            relayed += 1
        return relayed

    async def _mark_queued(self, event_key: str) -> None:
        marker = getattr(self._inbox, "mark_queued", None)
        if marker is not None:
            await marker(event_key)
