from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from whatsapp_support.domain import OutboundCommand, OutboundMode, OutboxState
from whatsapp_support.fakes import InMemoryOutboxStore
from whatsapp_support.postgres import PostgresDatabase


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def make_command(
    *, dedup_key: str = "cmd-1", recipient_id: str = "recipient-1"
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conv-1",
        recipient_id=recipient_id,
        mode=OutboundMode.FREEFORM,
        text="hi there",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=f"trace-{dedup_key}",
    )


@pytest_asyncio.fixture
async def postgres_db() -> PostgresDatabase:
    database = PostgresDatabase("sqlite+aiosqlite:///:memory:")
    await database.connect()
    await database.create_schema()
    try:
        yield database
    finally:
        await database.close()


@pytest_asyncio.fixture(params=["memory", "postgres"])
async def outbox_store(
    request: pytest.FixtureRequest,
    postgres_db: PostgresDatabase,
) -> InMemoryOutboxStore | object:
    if request.param == "memory":
        return InMemoryOutboxStore()
    return postgres_db.outbox


@pytest.mark.asyncio
async def test_claim_sets_finite_lease_and_blocks_unexpired_double_claim(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    now = utc_datetime(2026, 8, 22, 12, 0, 0)
    await store.enqueue(make_command(dedup_key="leased-once"))

    claimed = await store.claim_pending(now, 60)
    second_claim = await store.claim_pending(now + timedelta(seconds=30), 60)

    assert claimed is not None
    assert claimed.state is OutboxState.SENDING
    assert claimed.claimed_at == now
    assert claimed.lease_until == now + timedelta(seconds=60)
    assert claimed.attempts == 1
    assert second_claim is None


@pytest.mark.asyncio
async def test_expired_sending_lease_is_claimable_without_changing_dedup_key(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    first_claim_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    reclaim_at = first_claim_at + timedelta(seconds=61)
    await store.enqueue(make_command(dedup_key="stable-key"))
    first = await store.claim_pending(first_claim_at, 60)

    retry = await store.claim_pending(reclaim_at, 60)
    records = await store.records()

    assert first is not None
    assert retry is not None
    assert retry.command.dedup_key == "stable-key"
    assert retry.attempts == 2
    assert retry.claimed_at == reclaim_at
    assert retry.lease_until == reclaim_at + timedelta(seconds=60)
    assert records[0].last_error == "lease_expired"


@pytest.mark.asyncio
async def test_reclaim_expired_records_retryable_lease_expired_error(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    claimed_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    expired_at = claimed_at + timedelta(seconds=60)
    await store.enqueue(make_command(dedup_key="reclaim-me"))
    claimed = await store.claim_pending(claimed_at, 60)

    reclaimed = await store.reclaim_expired(expired_at)
    record = (await store.records())[0]

    assert claimed is not None
    assert reclaimed == 1
    assert record.state is OutboxState.RETRYABLE_FAILED
    assert record.claimed_at is None
    assert record.lease_until is None
    assert record.last_error == "lease_expired"
    assert record.command.dedup_key == "reclaim-me"


@pytest.mark.asyncio
async def test_reclaimed_record_retries_until_dead_letter_limit(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    await store.enqueue(make_command(dedup_key="dead-after-reclaim"))
    first = await store.claim_pending(utc_datetime(2026, 8, 22, 12, 0, 0), 60)
    assert first is not None
    assert await store.reclaim_expired(utc_datetime(2026, 8, 22, 12, 1, 0)) == 1

    second = await store.claim_pending(utc_datetime(2026, 8, 22, 12, 1, 1), 60)
    assert second is not None
    await store.mark_retryable("dead-after-reclaim", "timeout")
    third = await store.claim_pending(utc_datetime(2026, 8, 22, 12, 2, 0), 60)
    assert third is not None
    await store.mark_dead_letter("dead-after-reclaim", "timeout")

    record = (await store.records())[0]
    assert record.state is OutboxState.DEAD_LETTER
    assert record.attempts == 3
    assert record.lease_until is None
    assert await store.claim_pending(utc_datetime(2026, 8, 22, 12, 3, 0), 60) is None


@pytest.mark.asyncio
async def test_crash_recovery_never_claims_beyond_max_attempts(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    first_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    second_at = first_at + timedelta(minutes=1)
    exhausted_at = second_at + timedelta(minutes=1)
    await store.enqueue(make_command(dedup_key="crash-bounded"))

    first = await store.claim_pending(first_at, 30, max_attempts=2)
    second = await store.claim_pending(second_at, 30, max_attempts=2)
    exhausted = await store.claim_pending(exhausted_at, 30, max_attempts=2)

    assert first is not None and first.attempts == 1
    assert second is not None and second.attempts == 2
    assert exhausted is None
    record = (await store.records())[0]
    assert record.command.dedup_key == "crash-bounded"
    assert record.attempts == 2
    assert record.state is OutboxState.DEAD_LETTER
    assert record.last_error == "max_attempts_exhausted_after_lease_expiry"


@pytest.mark.asyncio
async def test_reclaimer_dead_letters_expired_final_attempt(
    outbox_store: InMemoryOutboxStore | object,
) -> None:
    store = outbox_store
    claimed_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    await store.enqueue(make_command(dedup_key="reclaim-bounded"))
    claimed = await store.claim_pending(claimed_at, 30, max_attempts=1)

    reclaimed = await store.reclaim_expired(
        claimed_at + timedelta(seconds=31), max_attempts=1
    )

    assert claimed is not None
    assert reclaimed == 1
    record = (await store.records())[0]
    assert record.state is OutboxState.DEAD_LETTER
    assert record.attempts == 1
    assert record.last_error == "max_attempts_exhausted_after_lease_expiry"


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["sent", "retryable", "dead_letter"])
async def test_stale_claim_completion_does_not_overwrite_newer_sending_attempt(
    outbox_store: InMemoryOutboxStore | object,
    completion: str,
) -> None:
    store = outbox_store
    first_claim_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    second_claim_at = utc_datetime(2026, 8, 22, 12, 1, 1)
    await store.enqueue(make_command(dedup_key=f"stale-{completion}"))
    stale = await store.claim_pending(first_claim_at, 60)
    assert stale is not None
    assert await store.reclaim_expired(first_claim_at + timedelta(seconds=60)) == 1
    current = await store.claim_pending(second_claim_at, 60)
    assert current is not None

    if completion == "sent":
        await store.mark_sent(
            stale.command.dedup_key,
            "wamid-late",
            attempts=stale.attempts,
            claimed_at=stale.claimed_at,
        )
    elif completion == "retryable":
        await store.mark_retryable(
            stale.command.dedup_key,
            "late-timeout",
            attempts=stale.attempts,
            claimed_at=stale.claimed_at,
        )
    else:
        await store.mark_dead_letter(
            stale.command.dedup_key,
            "late-timeout",
            attempts=stale.attempts,
            claimed_at=stale.claimed_at,
        )

    record = (await store.records())[0]
    assert record.state is OutboxState.SENDING
    assert record.attempts == 2
    assert record.claimed_at == second_claim_at
    assert record.lease_until == second_claim_at + timedelta(seconds=60)
    assert record.provider_message_id is None
    assert record.last_error == "lease_expired"
