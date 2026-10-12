from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

import whatsapp_support.meta_webhook as meta_webhook
from whatsapp_support.domain import EventType
from whatsapp_support.fakes import (
    FakeBusinessTools,
    FakeHandoff,
    FakeIdentity,
    FakeKnowledge,
    FakeModel,
    FakeTemplateCatalog,
    InMemoryConversationStore,
    InMemoryInboxStore,
    InMemoryOutboxStore,
    InMemoryQueue,
)
from whatsapp_support.worker import ConversationWorker
from whatsapp_support.meta_webhook import (
    MetaEventOutcome,
    MetaWebhookAdapter,
    MetaWebhookResult,
    parse_meta_events,
)


APP_SECRET = "test-meta-app-secret"
RECEIVED_AT = datetime(2026, 8, 23, 9, 30, tzinfo=UTC)


def meta_payload() -> dict[str, object]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "business-account-1",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "+15550000000",
                                "phone_number_id": "business-phone-1",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Customer"},
                                    "wa_id": "15551234567",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "15551234567",
                                    "id": "wamid-text-1",
                                    "timestamp": "1787446923",
                                    "type": "text",
                                    "text": {"body": "Where is my order?"},
                                },
                                {
                                    "from": "15551234567",
                                    "id": "wamid-image-1",
                                    "timestamp": "1787446924",
                                    "type": "image",
                                    "image": {"id": "media-1", "mime_type": "image/jpeg"},
                                },
                            ],
                            "statuses": [
                                {
                                    "id": "wamid-text-1",
                                    "recipient_id": "15551234567",
                                    "status": "delivered",
                                    "timestamp": "1787446925",
                                }
                            ],
                            "unrelated_provider_field": {"ignored": True},
                        },
                    }
                ],
            }
        ],
    }


def signed_meta_payload(payload: dict[str, object] | None = None) -> bytes:
    return json.dumps(
        meta_payload() if payload is None else payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def signature_for(raw_body: bytes) -> str:
    digest = hmac.new(APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def adapter(
    inbox: InMemoryInboxStore | None = None,
    queue: InMemoryQueue | None = None,
) -> MetaWebhookAdapter:
    return MetaWebhookAdapter(
        inbox or InMemoryInboxStore(),
        queue or InMemoryQueue(),
        APP_SECRET,
        clock=lambda: RECEIVED_AT,
    )


def unqueued(inbox: InMemoryInboxStore) -> list[object]:
    return asyncio.run(inbox.list_unqueued())


def test_parse_meta_text_and_unsupported_events_use_wa_id_and_utc() -> None:
    events = parse_meta_events(meta_payload(), RECEIVED_AT)

    assert [event.event_type for event in events] == [
        EventType.MESSAGE,
        EventType.UNSUPPORTED,
        EventType.STATUS,
    ]
    message = events[0]
    assert message.phone_number_id == "business-phone-1"
    assert message.conversation_id == "business-phone-1:15551234567"
    assert message.contact_id == "15551234567"
    assert message.recipient_id == "15551234567"
    assert message.provider_message_id == "wamid-text-1"
    assert message.occurred_at == datetime(2026, 8, 23, 1, 2, 3, tzinfo=UTC)
    assert message.received_at == RECEIVED_AT
    assert message.text == "Where is my order?"
    assert events[1].provider_message_id == "wamid-image-1"
    assert events[1].event_key == "business-phone-1:message:wamid-image-1"
    assert events[1].text is None
    assert events[2].provider_message_id == "wamid-text-1"
    assert events[2].occurred_at == datetime(2026, 8, 23, 1, 2, 5, tzinfo=UTC)


def test_invalid_signature_is_checked_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decode(_: object) -> object:
        raise AssertionError("JSON must not be decoded before HMAC verification")

    monkeypatch.setattr(meta_webhook.json, "loads", fail_decode)

    result = asyncio.run(adapter().receive(b"not-json", "sha256=bad"))

    assert result.status_code == 401
    assert result.accepted is False
    assert result.error_code == "invalid_signature"


def test_malformed_or_empty_meta_envelopes_return_bad_request() -> None:
    for payload in (
        {},
        {"object": "whatsapp_business_account", "entry": []},
        {"entry": [{"changes": [{"value": {"metadata": {}}}]}]},
    ):
        raw = signed_meta_payload(payload)
        result = asyncio.run(adapter().receive(raw, signature_for(raw)))

        assert result.status_code == 400
        assert result.accepted is False
        assert result.error_code == "invalid_payload"


def test_batch_dedup_and_partial_queue_failure_keep_unqueued_keys() -> None:
    inbox = InMemoryInboxStore()
    queue = InMemoryQueue(fail_next_publish=True)
    gateway = adapter(inbox, queue)
    raw = signed_meta_payload()

    result = asyncio.run(gateway.receive(raw, signature_for(raw)))

    assert result.status_code == 202
    assert result.accepted is True
    assert result.retryable is True
    assert result.unqueued_event_keys == (
        "business-phone-1:message:wamid-text-1",
    )
    assert [event.event_key for event in unqueued(inbox)] == [
        "business-phone-1:message:wamid-text-1"
    ]
    assert [outcome.queued for outcome in result.event_results] == [False, True, True]


def test_repeated_delivery_has_one_inbox_record_and_one_queue_delivery_per_event() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue()
        gateway = adapter(inbox, queue)
        raw = signed_meta_payload()
        signature = signature_for(raw)

        first = await gateway.receive(raw, signature)
        duplicate = await gateway.receive(raw, signature)

        assert first.status_code == 200
        assert duplicate.status_code == 200
        assert all(outcome.duplicate for outcome in duplicate.event_results)
        assert await queue.pending_delivery_ids() == [
            "delivery-1",
            "delivery-2",
            "delivery-3",
        ]
        assert await inbox.list_unqueued() == []

    asyncio.run(scenario())


def test_missing_provider_ids_use_canonical_sub_event_digest() -> None:
    first_payload = meta_payload()
    messages = first_payload["entry"][0]["changes"][0]["value"]["messages"]  # type: ignore[index]
    statuses = first_payload["entry"][0]["changes"][0]["value"]["statuses"]  # type: ignore[index]
    messages[0].pop("id")
    statuses[0].pop("id")

    second_payload = json.loads(signed_meta_payload(first_payload).decode("utf-8"))
    second_value = second_payload["entry"][0]["changes"][0]["value"]  # type: ignore[index]
    second_value["messages"] = [dict(reversed(list(messages[0].items()))), messages[1]]
    second_value["statuses"] = [dict(reversed(list(statuses[0].items())))]

    first_events = parse_meta_events(first_payload, RECEIVED_AT)
    second_events = parse_meta_events(second_payload, RECEIVED_AT)

    assert [event.event_key for event in first_events] == [event.event_key for event in second_events]
    assert first_events[0].event_key.startswith("business-phone-1:message:")
    assert first_events[2].event_key.startswith("business-phone-1:status:")
    assert first_events[0].payload_digest == second_events[0].payload_digest
    assert first_events[2].payload_digest == second_events[2].payload_digest


def test_second_delivery_retries_only_persisted_unqueued_event() -> None:
    async def scenario() -> None:
        inbox = InMemoryInboxStore()
        queue = InMemoryQueue(fail_next_publish=True)
        gateway = adapter(inbox, queue)
        raw = signed_meta_payload()
        signature = signature_for(raw)

        first = await gateway.receive(raw, signature)
        retry = await gateway.receive(raw, signature)

        assert first.status_code == 202
        assert retry.status_code == 200
        assert retry.retryable is False
        assert all(outcome.duplicate for outcome in retry.event_results)
        assert [outcome.queued for outcome in retry.event_results] == [
            True,
            False,
            False,
        ]
        assert await queue.pending_delivery_ids() == [
            "delivery-1",
            "delivery-2",
            "delivery-3",
        ]
        assert await inbox.list_unqueued() == []

    asyncio.run(scenario())


def test_status_and_unsupported_events_are_ignored_by_existing_worker() -> None:
    async def scenario() -> None:
        events = parse_meta_events(meta_payload(), RECEIVED_AT)
        conversations = InMemoryConversationStore()
        outbox = InMemoryOutboxStore()
        worker = ConversationWorker(
            conversations=conversations,
            outbox=outbox,
            tools=FakeBusinessTools(),
            identity=FakeIdentity(),
            knowledge=FakeKnowledge(),
            model=FakeModel(),
            templates=FakeTemplateCatalog(),
            handoff=FakeHandoff(),
            clock=lambda: RECEIVED_AT,
        )

        results = [await worker.process(event) for event in events[1:]]

        assert [result.status for result in results] == ["ignored", "ignored"]
        assert await outbox.records() == []

    asyncio.run(scenario())


def test_inbox_persistence_failure_returns_safe_server_error() -> None:
    class FailingInbox:
        async def record_once(self, event: object) -> bool:
            raise RuntimeError("database credentials must not leak")

    raw = signed_meta_payload()
    result = asyncio.run(
        MetaWebhookAdapter(FailingInbox(), InMemoryQueue(), APP_SECRET).receive(
            raw, signature_for(raw)
        )
    )

    assert result.status_code == 500
    assert result.accepted is False
    assert result.error_code == "persistence_failed"


def test_duplicate_lookup_persistence_failure_returns_safe_server_error() -> None:
    class FailingDuplicateLookupInbox:
        async def record_once(self, event: object) -> bool:
            return False

        async def list_unqueued(self) -> list[object]:
            raise RuntimeError("database credentials must not leak")

    raw = signed_meta_payload()
    result = asyncio.run(
        MetaWebhookAdapter(
            FailingDuplicateLookupInbox(), InMemoryQueue(), APP_SECRET
        ).receive(raw, signature_for(raw))
    )

    assert result.status_code == 500
    assert result.accepted is False
    assert result.error_code == "persistence_failed"
    assert "database credentials must not leak" not in repr(result)


def test_marker_persistence_failure_is_safe_and_keeps_event_unqueued() -> None:
    class FailingMarkerInbox(InMemoryInboxStore):
        async def mark_queued(self, event_key: str) -> None:
            raise RuntimeError("marker database credentials must not leak")

    inbox = FailingMarkerInbox()
    queue = InMemoryQueue()
    raw = signed_meta_payload()
    result = asyncio.run(
        MetaWebhookAdapter(inbox, queue, APP_SECRET).receive(
            raw, signature_for(raw)
        )
    )

    assert result.status_code == 500
    assert result.accepted is False
    assert result.error_code == "persistence_failed"
    assert result.event_results[0].queued is False
    assert result.event_results[0].error_code == "persistence_failed"
    assert result.unqueued_event_keys == (
        "business-phone-1:message:wamid-text-1",
    )
    assert [event.event_key for event in unqueued(inbox)] == [
        "business-phone-1:message:wamid-text-1"
    ]
    assert "marker database credentials must not leak" not in repr(result)


def test_result_contracts_are_frozen_dataclasses() -> None:
    outcome = MetaEventOutcome("event-1", False, True)
    result = MetaWebhookResult(200, True, False, (outcome,), ())

    with pytest.raises(FrozenInstanceError):
        outcome.queued = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status_code = 202  # type: ignore[misc]
