from __future__ import annotations

import json
from datetime import UTC, datetime

import fakeredis.aioredis
import pytest
import pytest_asyncio

from whatsapp_support.domain import EventType, InboundEvent
from whatsapp_support.redis_queue import RedisStreamQueue
from whatsapp_support.ports import QueueUnavailable


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_event(
    *, event_key: str = "evt-1", recipient_id: str = "recipient-1"
) -> InboundEvent:
    timestamp = utc_datetime(2026, 8, 22, 9, 0, 0)
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conv-1",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=timestamp,
        received_at=timestamp,
        payload_digest=f"digest-{event_key}",
        text="hello",
    )


@pytest_asyncio.fixture
async def redis_client() -> fakeredis.aioredis.FakeRedis:
    client: fakeredis.aioredis.FakeRedis = fakeredis.aioredis.FakeRedis(
        decode_responses=True
    )
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ensure_group_is_idempotent(redis_client: fakeredis.aioredis.FakeRedis) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")

    await queue.ensure_group()
    await queue.ensure_group()

    groups = await redis_client.xinfo_groups("stream")
    assert len(groups) == 1
    assert groups[0]["name"] == "workers"


@pytest.mark.asyncio
async def test_publish_and_receive_round_trip_uses_payload_mapping(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    event = make_event()
    await queue.ensure_group()

    await queue.publish(event)

    entries = await redis_client.xrange("stream")
    assert len(entries) == 1
    entry_id, fields = entries[0]
    assert list(fields) == ["payload"]
    assert json.loads(fields["payload"])["event_key"] == event.event_key

    delivery = await queue.receive(consumer="worker-a")
    assert delivery is not None
    assert delivery.delivery_id == entry_id
    assert delivery.event == event


@pytest.mark.asyncio
async def test_publish_translates_redis_failure_to_queue_unavailable() -> None:
    class FailingRedis:
        async def xadd(self, *args, **kwargs):
            del args, kwargs
            raise ConnectionError("redis unavailable")

    queue = RedisStreamQueue.from_client(FailingRedis(), "stream", "workers")

    with pytest.raises(QueueUnavailable) as excinfo:
        await queue.publish(make_event())

    assert isinstance(excinfo.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_receive_is_non_destructive_until_acknowledged(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    event = make_event()
    await queue.ensure_group()
    await queue.publish(event)

    first = await queue.receive(consumer="worker-a")
    assert first is not None
    assert await queue.receive(consumer="worker-a") is None

    pending_before_ack = await redis_client.xpending_range(
        "stream", "workers", "-", "+", 10
    )
    assert [item["message_id"] for item in pending_before_ack] == [first.delivery_id]

    await queue.acknowledge(first.delivery_id)

    pending_after_ack = await redis_client.xpending_range(
        "stream", "workers", "-", "+", 10
    )
    assert pending_after_ack == []
    assert await queue.receive(consumer="worker-a") is None


@pytest.mark.asyncio
async def test_unacknowledged_stream_delivery_can_be_reclaimed(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    await queue.ensure_group()
    await queue.publish(make_event())

    first = await queue.receive(consumer="worker-a")
    assert first is not None
    assert await queue.receive(consumer="worker-a") is None

    reclaimed = await queue.reclaim(consumer="worker-b", min_idle_ms=0)
    assert [item.event for item in reclaimed] == [first.event]
    assert [item.delivery_id for item in reclaimed] == [first.delivery_id]


@pytest.mark.asyncio
async def test_reclaim_drains_backlog_larger_than_one_pending_page(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    await queue.ensure_group()

    expected_keys: list[str] = []
    for index in range(120):
        event_key = f"evt-{index:03d}"
        expected_keys.append(event_key)
        await queue.publish(make_event(event_key=event_key))
        delivery = await queue.receive(consumer="worker-a")
        assert delivery is not None

    reclaimed = await queue.reclaim(consumer="worker-b", min_idle_ms=0)

    assert [item.event.event_key for item in reclaimed] == expected_keys
    pending = await redis_client.xpending_range("stream", "workers", "-", "+", 200)
    assert {item["consumer"] for item in pending} == {"worker-b"}


@pytest.mark.asyncio
async def test_receive_rejects_malformed_payload_via_event_mapping(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    await queue.ensure_group()
    await redis_client.xadd("stream", {"payload": json.dumps({"schema_version": 1})})

    with pytest.raises(ValueError, match=r"payload\."):
        await queue.receive(consumer="worker-a")


@pytest.mark.asyncio
async def test_receive_rejects_non_json_payload_text(
    redis_client: fakeredis.aioredis.FakeRedis,
) -> None:
    queue = RedisStreamQueue.from_client(redis_client, "stream", "workers")
    await queue.ensure_group()
    await redis_client.xadd("stream", {"payload": "not-json"})

    with pytest.raises(ValueError, match="payload must be valid JSON"):
        await queue.receive(consumer="worker-a")
