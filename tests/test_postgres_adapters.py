from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from whatsapp_support.domain import (
    AutomationState,
    CommitResult,
    Conversation,
    ConversationSummary,
    ConversationTurn,
    EventType,
    HandoffRequest,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    OutboxState,
)
from whatsapp_support.fakes import ConcurrentUpdateError
from whatsapp_support.memory import RollingSummaryPolicy, conservative_structured_summary
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


def make_event(
    *,
    event_key: str = "evt-1",
    conversation_id: str = "conv-1",
    occurred_at: datetime | None = None,
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    timestamp = occurred_at or utc_datetime(2026, 8, 22, 9, 0, 0)
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id=conversation_id,
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=timestamp,
        received_at=timestamp + timedelta(seconds=5),
        payload_digest=f"digest-{event_key}",
        text="hello",
    )


def make_conversation(
    *,
    conversation_id: str = "conv-1",
    version: int = 0,
    customer_last_inbound_at: datetime | None = None,
    automation_state: AutomationState = AutomationState.BOT_ACTIVE,
) -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        contact_id="contact-1",
        phone_number_id="phone-1",
        customer_last_inbound_at=customer_last_inbound_at,
        window_expires_at=(
            customer_last_inbound_at + timedelta(hours=24)
            if customer_last_inbound_at is not None
            else None
        ),
        automation_state=automation_state,
        version=version,
    )


def make_command(
    *,
    dedup_key: str = "cmd-1",
    conversation_id: str = "conv-1",
    recipient_id: str = "recipient-1",
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id=conversation_id,
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
async def db() -> PostgresDatabase:
    database = PostgresDatabase("sqlite+aiosqlite:///:memory:")
    await database.connect()
    await database.create_schema()
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_inbox_record_once_keeps_first_duplicate_and_lists_unqueued(
    db: PostgresDatabase,
) -> None:
    first = make_event(event_key="evt-dup")
    duplicate = make_event(
        event_key="evt-dup",
        occurred_at=utc_datetime(2026, 8, 22, 9, 5, 0),
    )

    assert await db.inbox.record_once(first) is True
    assert await db.inbox.record_once(duplicate) is False

    assert await db.inbox.get("evt-dup") == first
    assert await db.inbox.list_unqueued() == [first]


@pytest.mark.asyncio
async def test_inbox_relay_visibility_tracks_queued_and_processed_markers(
    db: PostgresDatabase,
) -> None:
    first = make_event(event_key="evt-1")
    second = make_event(event_key="evt-2")

    await db.inbox.record_once(first)
    await db.inbox.record_once(second)
    await db.inbox.mark_queued(first.event_key)

    assert await db.inbox.list_unqueued() == [second]

    await db.inbox.mark_processed(second.event_key)

    assert await db.inbox.list_unqueued() == []
    assert await db.inbox.get(first.event_key) == first
    assert await db.inbox.get(second.event_key) == second


@pytest.mark.asyncio
async def test_conversation_save_rejects_stale_version(
    db: PostgresDatabase,
) -> None:
    original = make_conversation(version=0)
    updated = make_conversation(
        version=1,
        customer_last_inbound_at=utc_datetime(2026, 8, 22, 10, 0, 0),
    )

    await db.conversations.save(original)
    await db.conversations.save(updated)

    with pytest.raises(ConcurrentUpdateError):
        await db.conversations.save(original)

    assert await db.conversations.get("conv-1") == updated


@pytest.mark.asyncio
async def test_conversation_save_is_strict_cas_for_first_insert_and_next_version(
    db: PostgresDatabase,
) -> None:
    original = make_conversation(version=0)
    await db.conversations.save(original)

    with pytest.raises(ConcurrentUpdateError):
        await db.conversations.save(original)
    with pytest.raises(ConcurrentUpdateError):
        await db.conversations.save(make_conversation(version=2))

    assert await db.conversations.get("conv-1") == original


@pytest.mark.asyncio
async def test_conversation_update_from_event_applies_domain_versioning(
    db: PostgresDatabase,
) -> None:
    event = make_event(occurred_at=utc_datetime(2026, 8, 22, 10, 0, 0))

    updated, changed = await db.conversations.update_from_event(event)
    unchanged, changed_again = await db.conversations.update_from_event(event)

    assert changed is True
    assert updated == make_conversation(
        version=1,
        customer_last_inbound_at=utc_datetime(2026, 8, 22, 10, 0, 0),
    )
    assert changed_again is False
    assert unchanged == updated


@pytest.mark.asyncio
async def test_outbox_enqueue_ignores_duplicate_dedup_key(
    db: PostgresDatabase,
) -> None:
    first = make_command(dedup_key="same-key")
    duplicate = make_command(dedup_key="same-key")

    await db.outbox.enqueue(first)
    await db.outbox.enqueue(duplicate)

    records = await db.outbox.records()
    assert len(records) == 1
    assert records[0].command == first
    assert records[0].state is OutboxState.PENDING
    assert records[0].attempts == 0


@pytest.mark.asyncio
async def test_outbox_state_transitions_and_expired_reclaim(
    db: PostgresDatabase,
) -> None:
    now = utc_datetime(2026, 8, 22, 11, 0, 0)
    command = make_command(dedup_key="retry-once")

    await db.outbox.enqueue(command)
    claimed = await db.outbox.claim_pending(now=now, lease_seconds=30)

    assert claimed is not None
    assert claimed.command == command
    assert claimed.state is OutboxState.SENDING
    assert claimed.attempts == 1
    assert claimed.claimed_at == now
    assert claimed.lease_until == now + timedelta(seconds=30)
    assert await db.outbox.claim_pending(now=now, lease_seconds=30) is None

    assert await db.outbox.reclaim_expired(now + timedelta(seconds=31)) == 1
    reclaimed = (await db.outbox.records())[0]
    assert reclaimed.state is OutboxState.RETRYABLE_FAILED
    assert reclaimed.claimed_at is None
    assert reclaimed.lease_until is None

    claimed_again = await db.outbox.claim_pending(
        now=now + timedelta(minutes=1), lease_seconds=30
    )
    assert claimed_again is not None
    assert claimed_again.attempts == 2

    await db.outbox.mark_sent("retry-once", "wamid-123")
    sent = (await db.outbox.records())[0]
    assert sent.state is OutboxState.SENT
    assert sent.provider_message_id == "wamid-123"
    assert sent.attempts == 2


@pytest.mark.asyncio
async def test_outbox_compatibility_claim_without_args_sets_finite_utc_lease(
    db: PostgresDatabase,
) -> None:
    await db.outbox.enqueue(make_command(dedup_key="compat-claim"))

    claimed = await db.outbox.claim_pending()

    assert claimed is not None
    assert claimed.claimed_at is not None
    assert claimed.claimed_at.tzinfo is not None
    assert claimed.claimed_at.utcoffset() == timedelta(0)
    assert claimed.lease_until is not None
    assert claimed.lease_until.tzinfo is not None
    assert claimed.lease_until.utcoffset() == timedelta(0)
    assert claimed.lease_until > claimed.claimed_at


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_seconds", [0, -1])
async def test_outbox_claim_rejects_explicit_non_positive_lease(
    db: PostgresDatabase,
    lease_seconds: int,
) -> None:
    await db.outbox.enqueue(make_command(dedup_key=f"bad-lease-{lease_seconds}"))

    with pytest.raises(ValueError, match="lease_seconds must be positive"):
        await db.outbox.claim_pending(lease_seconds=lease_seconds)


@pytest.mark.asyncio
async def test_transaction_commit_stores_conversation_outbox_and_processed_inbox(
    db: PostgresDatabase,
) -> None:
    event = make_event()
    conversation = make_conversation(
        version=1,
        customer_last_inbound_at=event.occurred_at,
    )
    command = make_command()
    await db.inbox.record_once(event)

    result = await db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=command,
    )

    assert result == CommitResult(status="committed")
    assert await db.conversations.get("conv-1") == conversation
    assert [record.command for record in await db.outbox.records()] == [command]
    assert await db.inbox.list_unqueued() == []
    assert await db.memory.get_recent_turns("conv-1", limit=6) == [
        ConversationTurn(
            customer_text="hello",
            assistant_text="hi there",
            event_key="evt-1",
            dedup_key="cmd-1",
        )
    ]


@pytest.mark.asyncio
async def test_transaction_replay_keeps_one_raw_turn_and_one_outbox_command(
    db: PostgresDatabase,
) -> None:
    event = make_event(event_key="evt-replay")
    conversation = make_conversation(
        version=1,
        customer_last_inbound_at=event.occurred_at,
    )
    command = make_command(dedup_key="reply:evt-replay")

    first = await db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=command,
    )
    replay = await db.transaction_store.commit_event(
        event=event,
        expected_version=1,
        conversation=conversation,
        command=command,
    )

    assert first.status == "committed"
    assert replay.status == "duplicate"
    assert await db.memory.get_recent_turns("conv-1", limit=6) == [
        ConversationTurn(
            customer_text="hello",
            assistant_text="hi there",
            event_key="evt-replay",
            dedup_key="reply:evt-replay",
        )
    ]
    assert [record.command for record in await db.outbox.records()] == [command]


@pytest.mark.asyncio
async def test_transaction_duplicate_outbox_still_commits_conversation_and_inbox(
    db: PostgresDatabase,
) -> None:
    event = make_event(event_key="evt-dedup")
    existing_command = make_command(dedup_key="cmd-dedup")
    duplicate_command = make_command(dedup_key="cmd-dedup")
    conversation = make_conversation(
        version=1,
        customer_last_inbound_at=event.occurred_at,
    )

    await db.inbox.record_once(event)
    await db.outbox.enqueue(existing_command)

    result = await db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=duplicate_command,
    )

    assert result == CommitResult(status="duplicate", reason="dedup_key_exists")
    assert await db.conversations.get("conv-1") == conversation
    assert await db.inbox.list_unqueued() == []
    assert [record.command for record in await db.outbox.records()] == [
        existing_command
    ]


@pytest.mark.asyncio
async def test_transaction_suppresses_supplied_outbox_for_human_active_state(
    db: PostgresDatabase,
) -> None:
    event = make_event(event_key="evt-human")
    conversation = make_conversation(
        version=1,
        automation_state=AutomationState.HUMAN_ACTIVE,
        customer_last_inbound_at=event.occurred_at,
    )

    result = await db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=make_command(dedup_key="must-not-send"),
    )

    assert result == CommitResult(status="committed")
    assert await db.conversations.get("conv-1") == conversation
    assert await db.outbox.records() == []


@pytest.mark.asyncio
async def test_transaction_replays_pending_handoff_until_marked_complete(
    db: PostgresDatabase,
) -> None:
    event = make_event(event_key="evt-handoff")
    conversation = make_conversation(
        version=1,
        automation_state=AutomationState.HUMAN_ACTIVE,
        customer_last_inbound_at=event.occurred_at,
    )
    request = HandoffRequest(
        request_id="handoff:evt-handoff",
        conversation_id=conversation.conversation_id,
        reason="unknown_intent",
        requested_by="conversation-worker",
        requested_at=event.occurred_at,
        note="trace_id=trace-handoff",
    )

    committed = await db.transaction_store.commit_event(
        event=event,
        expected_version=0,
        conversation=conversation,
        command=None,
        handoff_request=request,
    )
    replay = await db.transaction_store.commit_event(
        event=event,
        expected_version=1,
        conversation=conversation,
        command=None,
    )

    assert committed == CommitResult(status="committed", handoff_request=request)
    assert replay == CommitResult(
        status="duplicate",
        reason="event_already_processed",
        handoff_request=request,
    )

    await db.transaction_store.mark_handoff_complete(request.request_id)
    completed_replay = await db.transaction_store.commit_event(
        event=event,
        expected_version=1,
        conversation=conversation,
        command=None,
    )
    assert completed_replay == CommitResult(
        status="duplicate", reason="event_already_processed"
    )


@pytest.mark.asyncio
async def test_transaction_rollback_writes_neither_conversation_nor_outbox(
    db: PostgresDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_commit(session: AsyncSession) -> None:
        await session.flush()
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(AsyncSession, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        await db.transaction_store.commit_event(
            event=make_event(),
            expected_version=0,
            conversation=make_conversation(version=1),
            command=make_command(),
        )

    assert await db.conversations.get("conv-1") is None
    assert await db.outbox.records() == []
    assert await db.memory.raw_messages("conv-1") == []


@pytest.mark.asyncio
async def test_durable_summary_advances_and_survives_restart_with_raw_text_intact(
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "memory.sqlite3").as_posix()
    database_url = f"sqlite+aiosqlite:///{database_path}"
    database = PostgresDatabase(database_url)
    await database.create_schema()
    try:
        for index in range(8):
            event = make_event(
                event_key=f"evt-{index}",
                occurred_at=utc_datetime(2026, 8, 22, 9, index),
            )
            if index == 0:
                event = replace(
                    event, text="我的验证码是 123456，请帮我看退货政策"
                )
            conversation = make_conversation(
                version=index + 1,
                customer_last_inbound_at=event.occurred_at,
            )
            await database.transaction_store.commit_event(
                event=event,
                expected_version=index,
                conversation=conversation,
                command=make_command(
                    dedup_key=f"reply:evt-{index}",
                    conversation_id="conv-1",
                ),
            )
        first_summary = await database.memory.get_summary("conv-1")

        commandless = make_event(
            event_key="evt-commandless",
            occurred_at=utc_datetime(2026, 8, 22, 9, 8),
        )
        await database.transaction_store.commit_event(
            event=commandless,
            expected_version=8,
            conversation=make_conversation(
                version=9,
                customer_last_inbound_at=commandless.occurred_at,
            ),
            command=None,
        )
        commandless_summary = await database.memory.get_summary("conv-1")

        ninth = make_event(
            event_key="evt-8",
            occurred_at=utc_datetime(2026, 8, 22, 9, 9),
        )
        await database.transaction_store.commit_event(
            event=ninth,
            expected_version=9,
            conversation=make_conversation(
                version=10,
                customer_last_inbound_at=ninth.occurred_at,
            ),
            command=make_command(dedup_key="reply:evt-8"),
        )
        unchanged_summary = await database.memory.get_summary("conv-1")

        tenth = make_event(
            event_key="evt-9",
            occurred_at=utc_datetime(2026, 8, 22, 9, 10),
        )
        await database.transaction_store.commit_event(
            event=tenth,
            expected_version=10,
            conversation=make_conversation(
                version=11,
                customer_last_inbound_at=tenth.occurred_at,
            ),
            command=make_command(dedup_key="reply:evt-9"),
        )
        advanced_summary = await database.memory.get_summary("conv-1")
        raw_messages = await database.memory.raw_messages("conv-1")
    finally:
        await database.close()

    restarted = PostgresDatabase(database_url)
    await restarted.create_schema()
    try:
        persisted_summary = await restarted.memory.get_summary("conv-1")
    finally:
        await restarted.close()

    assert first_summary is not None
    assert commandless_summary == first_summary
    assert unchanged_summary == first_summary
    assert advanced_summary is not None
    assert advanced_summary.version > first_summary.version
    assert advanced_summary.covered_until > first_summary.covered_until
    assert persisted_summary == advanced_summary
    assert raw_messages[0]["text"] == "我的验证码是 123456，请帮我看退货政策"
    assert "123456" not in str(advanced_summary.data)


@pytest.mark.asyncio
async def test_durable_summary_redacts_credential_shapes_but_preserves_raw_text(
    tmp_path: Path,
) -> None:
    secrets = (
        "482913",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
        "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "AKIAIOSFODNN7EXAMPLE",
    )
    raw_text = " ".join(secrets)
    secret_top_level_key = "sk-proj-TopLevelSecretKey1234567890"
    secret_nested_key = "ghp_nestedsecretkey12345678901234567890"

    def summarize(
        conversation_id,
        turns,
        existing_summary,
        trace_id,
    ) -> ConversationSummary:
        del conversation_id, existing_summary, trace_id
        return ConversationSummary(
            version=1,
            covered_until=utc_datetime(2026, 8, 22, 9),
            data={
                "customer_goal": turns[-1].customer_text,
                "confirmed_entities": [],
                "customer_preferences": [],
                "unresolved_questions": [],
                "decisions": [],
                "handoff_state": "none",
                "open_items": {
                    "safe_item": "keep me",
                    secret_nested_key: "drop me",
                },
                secret_top_level_key: "drop me too",
            },
        )

    policy = RollingSummaryPolicy(
        max_recent_turns=0,
        summarize_at_turns=1,
        summarize_fn=summarize,
    )
    database = PostgresDatabase(
        f"sqlite+aiosqlite:///{(tmp_path / 'redacted-summary.sqlite3').as_posix()}",
        summary_policy=policy,
    )
    await database.create_schema()
    try:
        event = replace(make_event(event_key="evt-redaction"), text=raw_text)
        await database.transaction_store.commit_event(
            event=event,
            expected_version=0,
            conversation=make_conversation(
                version=1,
                customer_last_inbound_at=event.occurred_at,
            ),
            command=make_command(dedup_key="reply:evt-redaction"),
        )
        summary = await database.memory.get_summary("conv-1")
        raw_messages = await database.memory.raw_messages("conv-1")
    finally:
        await database.close()

    assert raw_messages[0]["text"] == raw_text
    assert summary is not None
    assert summary.data["customer_goal"] == "[redacted]"
    assert all(secret not in str(summary.data) for secret in secrets)
    assert secret_top_level_key not in summary.data
    assert secret_nested_key not in summary.data["open_items"]
    assert summary.data["open_items"] == {"safe_item": "keep me"}
    assert {
        "customer_goal",
        "confirmed_entities",
        "customer_preferences",
        "unresolved_questions",
        "decisions",
        "handoff_state",
    }.issubset(summary.data)


@pytest.mark.asyncio
async def test_summary_failure_cannot_undo_committed_reply(
    db: PostgresDatabase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def fail_summary(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("summary failed")

    monkeypatch.setattr(db.memory, "maybe_summarize", fail_summary, raising=False)
    event = make_event(event_key="evt-summary-failure")
    command = make_command(dedup_key="reply:evt-summary-failure")

    with caplog.at_level(logging.WARNING, logger="whatsapp_support.postgres"):
        result = await db.transaction_store.commit_event(
            event=event,
            expected_version=0,
            conversation=make_conversation(
                version=1,
                customer_last_inbound_at=event.occurred_at,
            ),
            command=command,
        )

    assert result.status == "committed"
    assert calls == 1
    assert [record.command for record in await db.outbox.records()] == [command]
    assert len(await db.memory.raw_messages("conv-1")) == 2
    assert [record.message for record in caplog.records] == [
        "conversation summary failed"
    ]
    assert caplog.records[0].conversation_id == "conv-1"
    assert caplog.records[0].trace_id == command.trace_id


@pytest.mark.asyncio
async def test_real_summary_generation_failure_logs_once_without_undoing_reply(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_summary(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("secret summary failure details")

    policy = RollingSummaryPolicy(
        max_recent_turns=1,
        summarize_at_turns=2,
        summarize_fn=fail_summary,
    )
    database = PostgresDatabase(
        f"sqlite+aiosqlite:///{(tmp_path / 'summary-failure.sqlite3').as_posix()}",
        summary_policy=policy,
    )
    await database.create_schema()
    try:
        with caplog.at_level(logging.WARNING):
            for index in range(2):
                event = make_event(
                    event_key=f"evt-real-summary-failure-{index}",
                    occurred_at=utc_datetime(2026, 8, 22, 9, index),
                )
                result = await database.transaction_store.commit_event(
                    event=event,
                    expected_version=index,
                    conversation=make_conversation(
                        version=index + 1,
                        customer_last_inbound_at=event.occurred_at,
                    ),
                    command=make_command(
                        dedup_key=f"reply:evt-real-summary-failure-{index}"
                    ),
                )

        warnings = [
            record
            for record in caplog.records
            if record.message == "conversation summary failed"
        ]
        assert result.status == "committed"
        assert len(await database.outbox.records()) == 2
        assert len(await database.memory.raw_messages("conv-1")) == 4
        assert await database.memory.get_summary("conv-1") is None
        assert len(warnings) == 1
        assert warnings[0].conversation_id == "conv-1"
        assert warnings[0].trace_id == "trace-reply:evt-real-summary-failure-1"
        assert "secret summary failure details" not in caplog.text
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_summary_cursor_covers_same_timestamp_turns_once_across_restart(
    tmp_path: Path,
) -> None:
    database_url = (
        f"sqlite+aiosqlite:///{(tmp_path / 'same-timestamp-summary.sqlite3').as_posix()}"
    )
    summarized_batches: list[list[str]] = []

    def summarize(conversation_id, turns, existing_summary, trace_id):
        summarized_batches.append([turn.event_key for turn in turns])
        return conservative_structured_summary(
            conversation_id, turns, existing_summary, trace_id
        )

    policy = RollingSummaryPolicy(
        max_recent_turns=2,
        summarize_at_turns=4,
        summarize_fn=summarize,
    )
    timestamp = utc_datetime(2026, 8, 22, 9, 0)

    async def append(database: PostgresDatabase, start: int, stop: int) -> None:
        for index in range(start, stop):
            event = make_event(event_key=f"evt-{index}", occurred_at=timestamp)
            await database.transaction_store.commit_event(
                event=event,
                expected_version=index,
                conversation=make_conversation(
                    version=index + 1,
                    customer_last_inbound_at=timestamp,
                ),
                command=make_command(dedup_key=f"reply:evt-{index}"),
            )

    database = PostgresDatabase(database_url, summary_policy=policy)
    await database.create_schema()
    try:
        await append(database, 0, 4)
        first = await database.memory.get_summary("conv-1")
    finally:
        await database.close()

    restarted = PostgresDatabase(database_url, summary_policy=policy)
    await restarted.create_schema()
    try:
        persisted = await restarted.memory.get_summary("conv-1")
        await append(restarted, 4, 8)
        final = await restarted.memory.get_summary("conv-1")
    finally:
        await restarted.close()

    assert first is not None
    assert first.covered_message_id is not None
    assert persisted == first
    assert final is not None
    assert final.covered_message_id > first.covered_message_id
    assert final.covered_until == first.covered_until == timestamp
    assert summarized_batches == [
        ["evt-0", "evt-1"],
        ["evt-2", "evt-3"],
        ["evt-4", "evt-5"],
    ]


@pytest.mark.asyncio
async def test_create_schema_migrates_legacy_summary_cursor_column(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-summary.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE conversation_summaries ("
            "conversation_id VARCHAR(255) PRIMARY KEY, "
            "version INTEGER NOT NULL, "
            "covered_until DATETIME NOT NULL, "
            "data JSON NOT NULL)"
        )

    database = PostgresDatabase(
        f"sqlite+aiosqlite:///{database_path.as_posix()}"
    )
    try:
        await database.create_schema()
        await database.memory.save_summary(
            "conv-legacy",
            conservative_structured_summary(
                "conv-legacy",
                [],
                None,
                "trace-legacy",
            ),
        )
    finally:
        await database.close()

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(conversation_summaries)")
        }
    assert "covered_message_id" in columns


@pytest.mark.asyncio
async def test_commandless_handoff_and_human_active_memory_survive_restart(
    tmp_path: Path,
) -> None:
    database_path = (tmp_path / "memory-cases.sqlite3").as_posix()
    database_url = f"sqlite+aiosqlite:///{database_path}"
    database = PostgresDatabase(database_url)
    await database.create_schema()
    cases = []
    try:
        for case_name in ("handoff", "human-active"):
            event = make_event(
                event_key=f"evt-{case_name}",
                conversation_id=f"conv-{case_name}",
            )
            conversation = make_conversation(
                conversation_id=event.conversation_id,
                version=1,
                customer_last_inbound_at=event.occurred_at,
                automation_state=(
                    AutomationState.HUMAN_ACTIVE
                    if case_name == "human-active"
                    else AutomationState.BOT_ACTIVE
                ),
            )
            handoff_request = (
                HandoffRequest(
                    request_id="handoff:memory-case",
                    conversation_id=event.conversation_id,
                    reason="unknown_intent",
                    requested_by="conversation-worker",
                    requested_at=event.occurred_at,
                    note="trace_id=trace-handoff-memory",
                )
                if case_name == "handoff"
                else None
            )
            supplied_command = (
                make_command(
                    dedup_key="reply:human-active",
                    conversation_id=event.conversation_id,
                )
                if case_name == "human-active"
                else None
            )

            committed = await database.transaction_store.commit_event(
                event=event,
                expected_version=0,
                conversation=conversation,
                command=supplied_command,
                handoff_request=handoff_request,
            )
            replay = await database.transaction_store.commit_event(
                event=event,
                expected_version=1,
                conversation=conversation,
                command=supplied_command,
                handoff_request=handoff_request,
            )
            cases.append((event, committed, replay))

            assert await database.memory.raw_messages(event.conversation_id) == [
                {
                    "role": "customer",
                    "text": "hello",
                    "event_key": event.event_key,
                    "outbox_dedup_key": None,
                }
            ]
        assert await database.outbox.records() == []
    finally:
        await database.close()

    restarted = PostgresDatabase(database_url)
    await restarted.create_schema()
    try:
        for event, committed, replay in cases:
            assert committed.status == "committed"
            assert replay.status == "duplicate"
            assert replay.reason == "event_already_processed"
            messages = await restarted.memory.raw_messages(event.conversation_id)
            assert len(messages) == 1
            assert messages[0]["role"] == "customer"
            assert messages[0]["event_key"] == event.event_key
            assert messages[0]["outbox_dedup_key"] is None
        assert await restarted.outbox.records() == []
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_summary_recovery_covers_oldest_unsummarized_turns_first(
    db: PostgresDatabase,
) -> None:
    def fail_summary(*args, **kwargs):
        raise RuntimeError("summary unavailable")

    db.memory._summary_policy.summarize_fn = fail_summary
    for index in range(12):
        event = make_event(
            event_key=f"evt-backlog-{index}",
            occurred_at=utc_datetime(2026, 8, 22, 9, index),
        )
        await db.transaction_store.commit_event(
            event=event,
            expected_version=index,
            conversation=make_conversation(
                version=index + 1,
                customer_last_inbound_at=event.occurred_at,
            ),
            command=make_command(dedup_key=f"reply:backlog-{index}"),
        )
    assert await db.memory.get_summary("conv-1") is None

    db.memory._summary_policy.summarize_fn = conservative_structured_summary
    recovery_event = make_event(
        event_key="evt-backlog-12",
        occurred_at=utc_datetime(2026, 8, 22, 9, 12),
    )
    await db.transaction_store.commit_event(
        event=recovery_event,
        expected_version=12,
        conversation=make_conversation(
            version=13,
            customer_last_inbound_at=recovery_event.occurred_at,
        ),
        command=make_command(dedup_key="reply:backlog-12"),
    )

    summary = await db.memory.get_summary("conv-1")
    assert summary is not None
    assert summary.covered_until == utc_datetime(2026, 8, 22, 9, 1)
