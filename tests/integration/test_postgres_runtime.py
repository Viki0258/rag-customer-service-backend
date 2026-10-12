from __future__ import annotations

import os
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from whatsapp_support.config import DATABASE_URL_ENV
from whatsapp_support.domain import (
    CommitResult,
    Conversation,
    EventType,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    OutboxState,
)
from whatsapp_support.postgres import PostgresDatabase


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


def make_event(
    *, suffix: str, occurred_at: datetime | None = None, recipient_id: str = "recipient-1"
) -> InboundEvent:
    timestamp = occurred_at or utc_datetime(2026, 8, 22, 9, 0, 0)
    return InboundEvent(
        event_key=f"evt-{suffix}",
        phone_number_id="phone-1",
        conversation_id=f"conv-{suffix}",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{suffix}",
        event_type=EventType.MESSAGE,
        occurred_at=timestamp,
        received_at=timestamp + timedelta(seconds=5),
        payload_digest=f"digest-{suffix}",
        text="hello",
    )


def make_conversation(*, suffix: str, version: int) -> Conversation:
    inbound_at = utc_datetime(2026, 8, 22, 9, 0, 0)
    return Conversation(
        conversation_id=f"conv-{suffix}",
        contact_id="contact-1",
        phone_number_id="phone-1",
        customer_last_inbound_at=inbound_at,
        window_expires_at=inbound_at + timedelta(hours=24),
        version=version,
    )


def make_command(*, suffix: str, recipient_id: str = "recipient-1") -> OutboundCommand:
    return OutboundCommand(
        dedup_key=f"cmd-{suffix}",
        conversation_id=f"conv-{suffix}",
        recipient_id=recipient_id,
        mode=OutboundMode.FREEFORM,
        text="hi there",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=f"trace-{suffix}",
    )


@pytest_asyncio.fixture
async def postgres_db() -> PostgresDatabase:
    database_url = os.getenv(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    database = PostgresDatabase(database_url)
    await database.connect()
    await database.create_schema()
    async with database.engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE TABLE "
                "inbox_events, conversations, outbox_commands, handoff_intents "
                "RESTART IDENTITY CASCADE"
            )
        )
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_real_postgres_inbox_duplicate_contract(
    postgres_db: PostgresDatabase,
) -> None:
    suffix = uuid4().hex
    event = make_event(suffix=suffix)

    assert await postgres_db.inbox.record_once(event) is True
    assert await postgres_db.inbox.record_once(event) is False
    assert await postgres_db.inbox.get(event.event_key) == event


@pytest.mark.asyncio
async def test_real_postgres_transaction_rollback_writes_no_partial_rows(
    postgres_db: PostgresDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    suffix = uuid4().hex

    async def fail_commit(session: AsyncSession) -> None:
        await session.flush()
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(AsyncSession, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        await postgres_db.transaction_store.commit_event(
            event=make_event(suffix=suffix),
            expected_version=0,
            conversation=make_conversation(suffix=suffix, version=1),
            command=make_command(suffix=suffix),
        )

    assert await postgres_db.conversations.get(f"conv-{suffix}") is None
    records = await postgres_db.outbox.records()
    assert all(record.command.dedup_key != f"cmd-{suffix}" for record in records)


@pytest.mark.asyncio
async def test_real_postgres_restart_reads_committed_rows(
    postgres_db: PostgresDatabase,
) -> None:
    database_url = os.environ[DATABASE_URL_ENV]
    suffix = uuid4().hex
    event = make_event(suffix=suffix)
    conversation = make_conversation(suffix=suffix, version=1)
    command = make_command(suffix=suffix)

    result = await postgres_db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=command,
    )
    await postgres_db.close()

    restarted = PostgresDatabase(database_url)
    await restarted.connect()
    try:
        assert result == CommitResult(status="committed")
        assert (
            await restarted.conversations.get(conversation.conversation_id)
            == conversation
        )
        assert command.dedup_key in {
            record.command.dedup_key for record in await restarted.outbox.records()
        }
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_real_postgres_two_worker_conversation_version_race(
    postgres_db: PostgresDatabase,
) -> None:
    suffix = uuid4().hex
    conversation_id = f"conv-{suffix}"
    first_event = make_event(suffix=f"{suffix}-a")
    second_event = make_event(suffix=f"{suffix}-b")
    first_event = replace(first_event, conversation_id=conversation_id)
    second_event = replace(second_event, conversation_id=conversation_id)
    first_conversation = replace(
        make_conversation(suffix=f"{suffix}-a", version=1),
        conversation_id=conversation_id,
    )
    second_conversation = replace(
        make_conversation(suffix=f"{suffix}-b", version=1),
        conversation_id=conversation_id,
    )
    await postgres_db.inbox.record_once(first_event)
    await postgres_db.inbox.record_once(second_event)

    first, second = await asyncio.gather(
        postgres_db.transaction_store.commit_event(
            first_event,
            0,
            first_conversation,
            make_command(suffix=f"{suffix}-a"),
        ),
        postgres_db.transaction_store.commit_event(
            second_event,
            0,
            second_conversation,
            make_command(suffix=f"{suffix}-b"),
        ),
    )

    assert sorted(result.status for result in (first, second)) == [
        "committed",
        "stale",
    ]
    records = await postgres_db.outbox.records()
    race_keys = {
        f"cmd-{suffix}-a",
        f"cmd-{suffix}-b",
    }
    assert len([record for record in records if record.command.dedup_key in race_keys]) == 1


@pytest.mark.asyncio
async def test_real_postgres_expired_final_lease_is_not_reclaimed_for_send(
    postgres_db: PostgresDatabase,
) -> None:
    suffix = uuid4().hex
    command = make_command(suffix=suffix)
    first_at = utc_datetime(2026, 8, 22, 12, 0, 0)
    await postgres_db.outbox.enqueue(command)

    first = await postgres_db.outbox.claim_pending(
        first_at, 30, max_attempts=1
    )
    reclaimed = await postgres_db.outbox.reclaim_expired(
        first_at + timedelta(seconds=31), max_attempts=1
    )
    retry = await postgres_db.outbox.claim_pending(
        first_at + timedelta(minutes=1), 30, max_attempts=1
    )

    assert first is not None and first.attempts == 1
    assert reclaimed == 1
    assert retry is None
    record = next(
        item
        for item in await postgres_db.outbox.records()
        if item.command.dedup_key == command.dedup_key
    )
    assert record.state is OutboxState.DEAD_LETTER
    assert record.attempts == 1


@pytest.mark.asyncio
async def test_real_postgres_clarification_survives_restart_and_is_not_repeated(
    postgres_db: PostgresDatabase,
) -> None:
    from whatsapp_support.clarification import QUESTIONS, pending_clarification
    from whatsapp_support.fakes import (
        FakeEmbedding, FakeHybridKnowledge, FakeModel, FakeIdentity,
        FakeTemplateCatalog, FakeHandoff,
    )
    from whatsapp_support.rag import RagAnswerService
    from whatsapp_support.worker import ConversationWorker

    suffix = uuid4().hex
    event = replace(make_event(suffix=suffix), text='那要几天？')
    command = replace(make_command(suffix=suffix), text=QUESTIONS['zh'],
                      model_name='deterministic-clarification-v1', prompt_version='test')
    committed = await postgres_db.transaction_store.commit_event(
        event, 0, make_conversation(suffix=suffix, version=1), command)
    assert committed.status == 'committed'
    await postgres_db.close()

    restarted = PostgresDatabase(os.environ[DATABASE_URL_ENV])
    await restarted.connect()
    try:
        turns = await restarted.memory.get_recent_turns(event.conversation_id, limit=6)
        assert pending_clarification(turns)
        embedding, model = FakeEmbedding(), FakeModel()
        now = event.occurred_at + timedelta(minutes=1)
        rag = RagAnswerService(embedding=embedding, knowledge=FakeHybridKnowledge(),
            model=model, memory=restarted.memory, prompt_version='test', retrieval_top_k=5,
            evidence_threshold=0, max_retrieved_chunks=5, max_reply_chars=500,
            clock=lambda: now)
        handoff = FakeHandoff()
        worker = ConversationWorker(restarted.conversations, restarted.outbox, None,
            FakeIdentity(), None, None, FakeTemplateCatalog(), handoff,
            transaction=restarted.transaction_store, rag_answer=rag, locale='zh-CN', clock=lambda: now)
        incoming = replace(event, event_key=f'next-{suffix}', provider_message_id=f'next-{suffix}',
                           text='还是不明白', occurred_at=now, received_at=now)
        result = await worker.process(incoming)
        assert result.status == 'handed_off'
        assert result.reason == 'clarification_requires_human'
        assert not embedding.calls and not model.calls
        rows = [r for r in await restarted.outbox.records()
                if r.command.conversation_id == event.conversation_id]
        assert len(rows) == 1 and rows[0].command.text == QUESTIONS['zh']
        await worker.process(incoming)
        rows = [r for r in await restarted.outbox.records()
                if r.command.conversation_id == event.conversation_id]
        assert len(rows) == 1
    finally:
        await restarted.close()
