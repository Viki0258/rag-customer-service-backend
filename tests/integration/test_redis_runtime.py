from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from whatsapp_support.config import REDIS_URL_ENV
from whatsapp_support.domain import EventType, InboundEvent
from whatsapp_support.redis_queue import RedisStreamQueue


pytestmark = pytest.mark.integration


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_event(*, suffix: str, recipient_id: str = "recipient-1") -> InboundEvent:
    timestamp = utc_datetime(2026, 8, 22, 9, 0, 0)
    return InboundEvent(
        event_key=f"evt-{suffix}",
        phone_number_id="phone-1",
        conversation_id=f"conv-{suffix}",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{suffix}",
        event_type=EventType.MESSAGE,
        occurred_at=timestamp,
        received_at=timestamp,
        payload_digest=f"digest-{suffix}",
        text="hello",
    )


@pytest.mark.asyncio
async def test_real_redis_stream_requires_ack_before_redelivery() -> None:
    redis_url = os.getenv(REDIS_URL_ENV)
    if not redis_url:
        pytest.skip(f"{REDIS_URL_ENV} is not configured")

    suffix = uuid4().hex
    stream_name = f"whatsapp-support:test:{suffix}"
    consumer_group = f"workers-{suffix}"
    queue = RedisStreamQueue(redis_url, stream_name, consumer_group)
    event = make_event(suffix=suffix)

    await queue.ensure_group()
    await queue.publish(event)

    first = await queue.receive(consumer="worker-a")
    assert first is not None
    assert await queue.receive(consumer="worker-a") is None

    reclaimed = await queue.reclaim(consumer="worker-b", min_idle_ms=0)
    assert [item.event.event_key for item in reclaimed] == [event.event_key]

    await queue.acknowledge(first.delivery_id)
    assert await queue.receive(consumer="worker-a") is None
