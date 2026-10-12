from __future__ import annotations

from whatsapp_support.domain import QueueDelivery
from whatsapp_support.ports import QueueConsumerPort
from whatsapp_support.worker import ConversationWorker, WorkerResult


class DurableEventProcessor:
    def __init__(
        self,
        worker: ConversationWorker,
        queue: QueueConsumerPort,
        *,
        consumer: str,
        reclaim_idle_ms: int = 60_000,
    ) -> None:
        self._worker = worker
        self._queue = queue
        self._consumer = consumer
        self._reclaim_idle_ms = reclaim_idle_ms

    async def process_once(self) -> WorkerResult | None:
        delivery = await self._queue.receive(consumer=self._consumer)
        if delivery is None:
            return None
        return await self._process_delivery(delivery)

    async def reclaim_once(self) -> int:
        deliveries = await self._queue.reclaim(
            consumer=self._consumer,
            min_idle_ms=self._reclaim_idle_ms,
        )
        processed = 0
        for delivery in deliveries:
            result = await self._process_delivery(delivery)
            if not result.retryable:
                processed += 1
        return processed

    async def _process_delivery(self, delivery: QueueDelivery) -> WorkerResult:
        result = await self._worker.process(delivery.event)
        if not result.retryable:
            await self._queue.acknowledge(delivery.delivery_id)
        return result
