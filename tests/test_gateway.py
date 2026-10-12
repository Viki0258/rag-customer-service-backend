from __future__ import annotations

import hashlib
import hmac
import json
import asyncio
from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import EventType
from whatsapp_support.fakes import InMemoryInboxStore, InMemoryQueue
from whatsapp_support.gateway import (
    ChallengeRejected,
    InboxRelay,
    WebhookGateway,
    normalize_event,
    verify_challenge,
    verify_signature,
)


APP_SECRET = "test-app-secret"
RECEIVED_AT = datetime(2026, 8, 22, 9, 30, tzinfo=UTC)


def payload(*, provider_message_id: str | None = "wamid-1") -> dict[str, object]:
    event: dict[str, object] = {
        "phone_number_id": "phone-1",
        "conversation_id": "conversation-1",
        "contact_id": "contact-1",
        "recipient_id": "recipient-1",
        "event_type": "message",
        "occurred_at": "2026-08-22T09:00:00Z",
        "text": "Where is my order?",
        "identity_proof": {"kind": "otp", "value": "123456"},
    }
    if provider_message_id is not None:
        event["provider_message_id"] = provider_message_id
    return event


def raw_payload(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def signature_for(raw_body: bytes, secret: str = APP_SECRET) -> str:
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_verify_signature_accepts_exact_sha256_hmac() -> None:
    raw_body = b'{"message":"hello"}'

    assert verify_signature(raw_body, signature_for(raw_body), APP_SECRET) is True


def test_verify_signature_rejects_malformed_or_wrong_headers_without_raising() -> None:
    raw_body = b'{"message":"hello"}'

    for header in (None, "", "sha512=abcd", "sha256=not-hex", "sha256="):
        assert verify_signature(raw_body, header, APP_SECRET) is False
    assert verify_signature(raw_body, signature_for(raw_body, "wrong"), APP_SECRET) is False


def test_verify_challenge_requires_subscribe_and_exact_token() -> None:
    assert verify_challenge("subscribe", "verify-me", "challenge", "verify-me") == "challenge"

    with pytest.raises(ChallengeRejected):
        verify_challenge("unsubscribe", "verify-me", "challenge", "verify-me")
    with pytest.raises(ChallengeRejected):
        verify_challenge("subscribe", "wrong", "challenge", "verify-me")


def test_normalize_event_uses_provider_id_and_utc_timestamps() -> None:
    event = normalize_event(payload(), RECEIVED_AT)

    assert event.event_key == "phone-1:message:wamid-1"
    assert event.event_type is EventType.MESSAGE
    assert event.occurred_at == datetime(2026, 8, 22, 9, tzinfo=UTC)
    assert event.received_at == RECEIVED_AT
    assert event.identity_proof is not None
    assert event.identity_proof.value == "123456"


def test_normalize_event_fallback_key_is_deterministic_for_canonical_payload() -> None:
    first = payload(provider_message_id=None)
    second = dict(reversed(list(first.items())))

    first_event = normalize_event(first, RECEIVED_AT)
    second_event = normalize_event(second, RECEIVED_AT)

    expected_digest = hashlib.sha256(raw_payload(first)).hexdigest()
    assert first_event.event_key == f"phone-1:message:{expected_digest}"
    assert second_event.event_key == first_event.event_key
    assert first_event.payload_digest == expected_digest


def test_normalize_event_rejects_non_utc_or_unknown_event_types() -> None:
    non_utc = payload()
    non_utc["occurred_at"] = "2026-08-22T17:00:00+08:00"
    with pytest.raises(ValueError):
        normalize_event(non_utc, RECEIVED_AT)

    unknown = payload()
    unknown["event_type"] = "reaction"
    with pytest.raises(ValueError):
        normalize_event(unknown, RECEIVED_AT)


def test_gateway_rejects_malformed_payload_after_signature_verification() -> None:
    inbox = InMemoryInboxStore()
    queue = InMemoryQueue()
    gateway = WebhookGateway(inbox, queue, APP_SECRET)
    raw_body = b'{"event_type":"message"}'

    result = asyncio.run(gateway.receive(raw_body, signature_for(raw_body)))

    assert result.status_code == 400
    assert result.accepted is False
    assert result.error_code == "invalid_payload"
    assert asyncio.run(inbox.list_unqueued()) == []


def test_gateway_rejects_invalid_signature_before_parsing_payload() -> None:
    inbox = InMemoryInboxStore()
    queue = InMemoryQueue()
    gateway = WebhookGateway(inbox, queue, APP_SECRET)

    result = asyncio.run(
        gateway.receive(b"not-json", "sha256=bad")
    )

    assert result.status_code == 401
    assert result.accepted is False
    assert result.error_code == "invalid_signature"


def test_gateway_deduplicates_repeated_webhooks_and_publishes_once() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue()
        gateway = WebhookGateway(inbox, queue, APP_SECRET)
        raw_body = raw_payload(payload())
        signature = signature_for(raw_body)

        first = await gateway.receive(raw_body, signature)
        duplicate = await gateway.receive(raw_body, signature)

        assert first.status_code == 200
        assert first.accepted is True
        assert first.duplicate is False
        assert first.queued is True
        assert duplicate.status_code == 200
        assert duplicate.accepted is True
        assert duplicate.duplicate is True
        assert duplicate.queued is False
        assert await queue.pending_delivery_ids() == ["delivery-1"]
        assert await inbox.list_unqueued() == []

    asyncio.run(scenario())


def test_gateway_accepts_explicit_unsupported_event_without_robot_side_effects() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue()
        gateway = WebhookGateway(inbox, queue, APP_SECRET)
        unsupported = payload()
        unsupported["event_type"] = "unsupported"
        unsupported["provider_message_id"] = "unsupported-1"
        raw_body = raw_payload(unsupported)

        result = await gateway.receive(raw_body, signature_for(raw_body))

        assert result.status_code == 200
        assert result.accepted is True
        assert result.queued is True
        expected_digest = hashlib.sha256(raw_body).hexdigest()
        assert result.event_key == f"phone-1:unsupported:{expected_digest}"

    asyncio.run(scenario())


def test_gateway_returns_accepted_202_when_queue_fails_after_persistence() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue(fail_next_publish=True)
        gateway = WebhookGateway(inbox, queue, APP_SECRET)
        raw_body = raw_payload(payload())

        result = await gateway.receive(raw_body, signature_for(raw_body))

        assert result.status_code == 202
        assert result.accepted is True
        assert result.duplicate is False
        assert result.queued is False
        assert await inbox.list_unqueued()
        assert await queue.pending_delivery_ids() == []

    asyncio.run(scenario())


def test_provider_retry_republishes_persisted_unqueued_event() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue(fail_next_publish=True)
        gateway = WebhookGateway(inbox, queue, APP_SECRET)
        raw_body = raw_payload(payload())
        signature = signature_for(raw_body)

        first = await gateway.receive(raw_body, signature)
        retry = await gateway.receive(raw_body, signature)

        assert first.status_code == 202
        assert retry.status_code == 200
        assert retry.duplicate is True
        assert retry.queued is True
        assert await inbox.list_unqueued() == []
        assert await queue.pending_delivery_ids() == ["delivery-1"]

    asyncio.run(scenario())


def test_inbox_relay_recovers_persisted_event_and_marks_only_successes() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue()
        event = normalize_event(payload(), RECEIVED_AT)
        await inbox.record_once(event)
        queue.fail_next_publish = True
        relay = InboxRelay(inbox, queue)

        failed_count = await relay.relay_once()
        assert failed_count == 0
        assert await inbox.list_unqueued() == [event]

        relayed_count = await relay.relay_once()
        assert relayed_count == 1
        assert await inbox.list_unqueued() == []
        assert await queue.pending_delivery_ids() == ["delivery-1"]

    asyncio.run(scenario())
