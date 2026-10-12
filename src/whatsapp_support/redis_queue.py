from __future__ import annotations

import json
from typing import Any

from whatsapp_support.domain import InboundEvent, QueueDelivery
from whatsapp_support.ports import QueueUnavailable
from whatsapp_support.serialization import event_from_mapping, event_to_mapping


class RedisStreamQueue:
    def __init__(
        self,
        redis_url: str,
        stream_name: str,
        consumer_group: str,
        *,
        _client: Any | None = None,
    ) -> None:
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        if _client is None:
            from redis.asyncio import Redis

            self._client: Any = Redis.from_url(redis_url, decode_responses=True)
        else:
            self._client = _client

    @classmethod
    def from_client(
        cls, client: Any, stream_name: str, consumer_group: str
    ) -> "RedisStreamQueue":
        return cls("", stream_name, consumer_group, _client=client)

    async def ensure_group(self) -> None:
        from redis.exceptions import ResponseError

        try:
            await self._client.xgroup_create(
                self._stream_name,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, event: InboundEvent) -> None:
        payload = json.dumps(event_to_mapping(event), separators=(",", ":"))
        try:
            await self._client.xadd(self._stream_name, {"payload": payload})
        except Exception as exc:
            raise QueueUnavailable("Redis stream publish failed") from exc

    async def close(self) -> None:
        closer = getattr(self._client, "aclose", None)
        if closer is not None:
            await closer()

    async def receive(
        self, *, consumer: str, block_ms: int = 0
    ) -> QueueDelivery | None:
        response = await self._client.xreadgroup(
            groupname=self._consumer_group,
            consumername=consumer,
            streams={self._stream_name: ">"},
            count=1,
            block=block_ms if block_ms > 0 else None,
        )
        if not response:
            return None
        _, entries = response[0]
        if not entries:
            return None
        return self._decode_delivery(entries[0])

    async def reclaim(
        self, *, consumer: str, min_idle_ms: int
    ) -> list[QueueDelivery]:
        reclaimed: list[QueueDelivery] = []
        cursor = "-"

        while True:
            pending = await self._pending_range(cursor=cursor, min_idle_ms=min_idle_ms)
            if cursor != "-":
                pending = [
                    item
                    for item in pending
                    if self._message_id_key(self._pending_message_id(item))
                    > self._message_id_key(cursor)
                ]
            message_ids = [self._pending_message_id(item) for item in pending]
            if not message_ids:
                return reclaimed
            claimed = await self._client.xclaim(
                self._stream_name,
                self._consumer_group,
                consumer,
                min_idle_ms,
                message_ids,
            )
            reclaimed.extend(self._decode_delivery(entry) for entry in claimed)
            cursor = message_ids[-1]

    async def acknowledge(self, delivery_id: str) -> None:
        await self._client.xack(
            self._stream_name,
            self._consumer_group,
            delivery_id,
        )

    def _decode_delivery(self, entry: Any) -> QueueDelivery:
        delivery_id, fields = entry
        payload_raw = fields.get("payload")
        if isinstance(payload_raw, bytes):
            payload_text = payload_raw.decode("utf-8")
        elif isinstance(payload_raw, str):
            payload_text = payload_raw
        else:
            raise ValueError("payload.payload must be a string")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError("payload must be valid JSON") from exc
        event = event_from_mapping(payload)
        return QueueDelivery(delivery_id=str(delivery_id), event=event)

    async def _pending_range(self, *, cursor: str, min_idle_ms: int) -> list[Any]:
        kwargs: dict[str, Any] = {}
        if min_idle_ms > 0:
            kwargs["idle"] = min_idle_ms
        return await self._client.xpending_range(
            self._stream_name,
            self._consumer_group,
            cursor,
            "+",
            100,
            **kwargs,
        )

    @staticmethod
    def _pending_message_id(item: Any) -> str:
        if isinstance(item, dict):
            message_id = item.get("message_id")
        else:
            message_id = item[0]
        if isinstance(message_id, bytes):
            return message_id.decode("utf-8")
        return str(message_id)

    @staticmethod
    def _message_id_key(message_id: str) -> tuple[int, int]:
        milliseconds, sequence = message_id.split("-", maxsplit=1)
        return int(milliseconds), int(sequence)
