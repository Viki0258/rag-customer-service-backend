from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    and_,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column
from sqlalchemy.pool import StaticPool

from whatsapp_support.domain import (
    AutomationState,
    CommitResult,
    Conversation,
    ConversationSummary,
    ConversationTurn,
    HandoffRequest,
    InboundEvent,
    OutboundCommand,
    OutboxRecord,
    OutboxState,
)
from whatsapp_support.ports import ConcurrentUpdateError
from whatsapp_support.memory import (
    RollingSummaryPolicy,
    conservative_structured_summary,
)
from whatsapp_support.serialization import (
    command_to_mapping,
    conversation_from_mapping,
    conversation_to_mapping,
    event_from_mapping,
    event_to_mapping,
    outbox_record_from_mapping,
)


logger = logging.getLogger(__name__)


DEFAULT_OUTBOX_LEASE_SECONDS = 60


class Base(DeclarativeBase):
    pass


class InboxRow(Base):
    __tablename__ = "inbox_events"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_inbox_events_event_key"),
        Index("ix_inbox_events_queue_status", "queued_at", "processed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(String(255), nullable=False)
    event_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationRow(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_conversation_id", "conversation_id"),)

    conversation_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class OutboxRow(Base):
    __tablename__ = "outbox_commands"
    __table_args__ = (
        UniqueConstraint("dedup_key", name="uq_outbox_commands_dedup_key"),
        Index("ix_outbox_commands_state", "state"),
        Index("ix_outbox_commands_lease_until", "lease_until"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dedup_key: Mapped[str] = mapped_column(String(255), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    command_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state: Mapped[str] = mapped_column(String(64), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(2000))


class HandoffIntentRow(Base):
    __tablename__ = "handoff_intents"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_handoff_intents_request_id"),
        UniqueConstraint("event_key", name="uq_handoff_intents_event_key"),
        Index("ix_handoff_intents_pending", "completed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationMessageRow(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_conversation_messages_event_key"),
        UniqueConstraint(
            "outbox_dedup_key",
            name="uq_conversation_messages_outbox_dedup_key",
        ),
        Index(
            "ix_conversation_messages_conversation",
            "conversation_id",
            "occurred_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_key: Mapped[str | None] = mapped_column(String(255))
    inbound_event_key: Mapped[str | None] = mapped_column(String(255))
    outbox_dedup_key: Mapped[str | None] = mapped_column(String(255))


class ConversationSummaryRow(Base):
    __tablename__ = "conversation_summaries"

    conversation_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    covered_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    covered_message_id: Mapped[int | None] = mapped_column(Integer)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class PostgresDatabase:
    def __init__(
        self,
        database_url: str,
        *,
        summary_policy: RollingSummaryPolicy | None = None,
    ) -> None:
        self.database_url = _normalize_database_url(database_url)
        self.engine = create_async_engine(
            self.database_url,
            pool_pre_ping=True,
            **_engine_options(self.database_url),
        )
        self._session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self.inbox = PostgresInboxStore(self._session_factory, self.engine)
        self.conversations = PostgresConversationStore(
            self._session_factory, self.engine
        )
        self.outbox = PostgresOutboxStore(self._session_factory, self.engine)
        self.memory = PostgresConversationMemory(
            self._session_factory,
            self.engine,
            summary_policy=summary_policy,
        )
        self.transaction_store = PostgresConversationOutboxTransaction(
            self._session_factory, self.engine, memory=self.memory
        )

    async def connect(self) -> None:
        async with self.engine.connect():
            pass

    async def close(self) -> None:
        await self.engine.dispose()

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            if self.engine.dialect.name == "postgresql":
                await connection.execute(
                    text(
                        "ALTER TABLE conversation_summaries "
                        "ADD COLUMN IF NOT EXISTS covered_message_id INTEGER"
                    )
                )
            else:
                columns = await connection.run_sync(
                    lambda sync_connection: {
                        column["name"]
                        for column in inspect(sync_connection).get_columns(
                            ConversationSummaryRow.__tablename__
                        )
                    }
                )
                if "covered_message_id" not in columns:
                    await connection.execute(
                        text(
                            "ALTER TABLE conversation_summaries "
                            "ADD COLUMN covered_message_id INTEGER"
                        )
                    )


class PostgresInboxStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    async def record_once(self, event: InboundEvent) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                _insert_ignore(
                    self._engine,
                    InboxRow,
                    conflict_columns=["event_key"],
                    values={
                        "event_key": event.event_key,
                        "event_payload": event_to_mapping(event),
                        "received_at": event.received_at,
                    },
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return result.rowcount == 1

    async def get(self, event_key: str) -> InboundEvent | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(InboxRow).where(InboxRow.event_key == event_key)
            )
        if row is None:
            return None
        return event_from_mapping(row.event_payload)

    async def list_unqueued(self) -> list[InboundEvent]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(InboxRow)
                    .where(InboxRow.queued_at.is_(None))
                    .where(InboxRow.processed_at.is_(None))
                    .order_by(InboxRow.id)
                )
            ).all()
        return [event_from_mapping(row.event_payload) for row in rows]

    async def mark_queued(self, event_key: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(InboxRow)
                .where(InboxRow.event_key == event_key)
                .where(InboxRow.processed_at.is_(None))
                .values(queued_at=_utc_now())
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def mark_processed(self, event_key: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(InboxRow)
                .where(InboxRow.event_key == event_key)
                .values(queued_at=None, processed_at=_utc_now())
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class PostgresConversationStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._session_factory() as session:
            row = await session.get(ConversationRow, conversation_id)
        if row is None:
            return None
        return conversation_from_mapping(row.payload)

    async def save(self, conversation: Conversation) -> None:
        async with self._session_factory() as session:
            if conversation.version == 0:
                result = await session.execute(
                    _insert_conversation_ignore(self._engine, conversation)
                )
            else:
                result = await session.execute(
                    update(ConversationRow)
                    .where(
                        ConversationRow.conversation_id
                        == conversation.conversation_id
                    )
                    .where(ConversationRow.version == conversation.version - 1)
                    .values(
                        payload=conversation_to_mapping(conversation),
                        version=conversation.version,
                    )
                )
            if result.rowcount != 1:
                await session.rollback()
                raise ConcurrentUpdateError(
                    f"stale conversation version for {conversation.conversation_id}"
                )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def update_from_event(
        self, event: InboundEvent
    ) -> tuple[Conversation, bool]:
        async with self._session_factory() as session:
            row = await session.scalar(
                _for_update(
                    select(ConversationRow).where(
                        ConversationRow.conversation_id == event.conversation_id
                    ),
                    self._engine,
                )
            )
            current = (
                conversation_from_mapping(row.payload)
                if row is not None
                else Conversation(
                    conversation_id=event.conversation_id,
                    contact_id=event.contact_id,
                    phone_number_id=event.phone_number_id,
                )
            )
            updated, changed = current.apply_inbound(event)
            if changed:
                if row is None:
                    session.add(_conversation_row(updated))
                else:
                    row.payload = conversation_to_mapping(updated)
                    row.version = updated.version
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return updated, changed


class PostgresOutboxStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    async def enqueue(self, command: OutboundCommand) -> None:
        async with self._session_factory() as session:
            await session.execute(_insert_outbox_ignore(self._engine, command))
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def claim_pending(
        self,
        now: datetime | None = None,
        lease_seconds: int | None = None,
        max_attempts: int | None = None,
    ) -> OutboxRecord | None:
        claim_time = _utc_now() if now is None else _ensure_utc(now)
        effective_lease_seconds = _effective_lease_seconds(lease_seconds)
        _validate_max_attempts(max_attempts)
        async with self._session_factory() as session:
            if max_attempts is not None:
                await session.execute(
                    update(OutboxRow)
                    .where(OutboxRow.attempts >= max_attempts)
                    .where(
                        or_(
                            OutboxRow.state.in_(
                                [
                                    OutboxState.PENDING.value,
                                    OutboxState.RETRYABLE_FAILED.value,
                                ]
                            ),
                            (
                                (OutboxRow.state == OutboxState.SENDING.value)
                                & OutboxRow.lease_until.is_not(None)
                                & (OutboxRow.lease_until <= claim_time)
                            ),
                        )
                    )
                    .values(
                        state=OutboxState.DEAD_LETTER.value,
                        claimed_at=None,
                        lease_until=None,
                        last_error="max_attempts_exhausted_after_lease_expiry",
                    )
                )
            statement = (
                select(OutboxRow)
                .where(
                    or_(
                        OutboxRow.state.in_(
                            [
                                OutboxState.PENDING.value,
                                OutboxState.RETRYABLE_FAILED.value,
                            ]
                        ),
                        (
                            (OutboxRow.state == OutboxState.SENDING.value)
                            & OutboxRow.lease_until.is_not(None)
                            & (OutboxRow.lease_until <= claim_time)
                        ),
                    )
                )
                .order_by(OutboxRow.id)
                .limit(1)
            )
            if max_attempts is not None:
                statement = statement.where(OutboxRow.attempts < max_attempts)
            row = await session.scalar(
                _for_update(
                    statement,
                    self._engine,
                    skip_locked=True,
                )
            )
            if row is None:
                await session.commit()
                return None
            expired_lease = row.state == OutboxState.SENDING.value
            row.state = OutboxState.SENDING.value
            row.attempts += 1
            row.claimed_at = claim_time
            row.lease_until = claim_time + timedelta(seconds=effective_lease_seconds)
            if expired_lease:
                row.last_error = "lease_expired"
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return _outbox_row_to_record(row)

    async def mark_sent(
        self,
        dedup_key: str,
        provider_message_id: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        await self._update_state(
            dedup_key,
            state=OutboxState.SENT,
            provider_message_id=provider_message_id,
            last_error=None,
            attempts=attempts,
            claimed_at=claimed_at,
        )

    async def mark_retryable(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        await self._update_state(
            dedup_key,
            state=OutboxState.RETRYABLE_FAILED,
            last_error=reason,
            attempts=attempts,
            claimed_at=claimed_at,
        )

    async def mark_dead_letter(
        self,
        dedup_key: str,
        reason: str,
        *,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        await self._update_state(
            dedup_key,
            state=OutboxState.DEAD_LETTER,
            last_error=reason,
            attempts=attempts,
            claimed_at=claimed_at,
        )

    async def reclaim_expired(
        self, now: datetime, max_attempts: int | None = None
    ) -> int:
        reclaim_time = _ensure_utc(now)
        _validate_max_attempts(max_attempts)
        async with self._session_factory() as session:
            exhausted_count = 0
            if max_attempts is not None:
                exhausted = await session.execute(
                    update(OutboxRow)
                    .where(OutboxRow.state == OutboxState.SENDING.value)
                    .where(OutboxRow.lease_until.is_not(None))
                    .where(OutboxRow.lease_until <= reclaim_time)
                    .where(OutboxRow.attempts >= max_attempts)
                    .values(
                        state=OutboxState.DEAD_LETTER.value,
                        claimed_at=None,
                        lease_until=None,
                        last_error="max_attempts_exhausted_after_lease_expiry",
                    )
                )
                exhausted_count = exhausted.rowcount
            retryable_statement = (
                update(OutboxRow)
                .where(OutboxRow.state == OutboxState.SENDING.value)
                .where(OutboxRow.lease_until.is_not(None))
                .where(OutboxRow.lease_until <= reclaim_time)
            )
            if max_attempts is not None:
                retryable_statement = retryable_statement.where(
                    OutboxRow.attempts < max_attempts
                )
            retryable = await session.execute(
                retryable_statement.values(
                    state=OutboxState.RETRYABLE_FAILED.value,
                    claimed_at=None,
                    lease_until=None,
                    last_error="lease_expired",
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return exhausted_count + retryable.rowcount

    async def records(self) -> list[OutboxRecord]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(select(OutboxRow).order_by(OutboxRow.id))
            ).all()
        return [_outbox_row_to_record(row) for row in rows]

    async def _update_state(
        self,
        dedup_key: str,
        *,
        state: OutboxState,
        provider_message_id: str | None = None,
        last_error: str | None,
        attempts: int | None = None,
        claimed_at: datetime | None = None,
    ) -> None:
        statement = update(OutboxRow).where(OutboxRow.dedup_key == dedup_key)
        if attempts is not None or claimed_at is not None:
            statement = statement.where(OutboxRow.state == OutboxState.SENDING.value)
            if attempts is not None:
                statement = statement.where(OutboxRow.attempts == attempts)
            if claimed_at is not None:
                statement = statement.where(OutboxRow.claimed_at == _ensure_utc(claimed_at))
        async with self._session_factory() as session:
            await session.execute(
                statement.values(
                    state=state.value,
                    provider_message_id=provider_message_id,
                    claimed_at=None,
                    lease_until=None,
                    last_error=last_error,
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class PostgresConversationMemory:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
        *,
        summary_policy: RollingSummaryPolicy | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self._summary_policy = summary_policy or RollingSummaryPolicy(
            summarize_fn=conservative_structured_summary
        )

    async def get_recent_turns(
        self, conversation_id: str, *, limit: int
    ) -> list[ConversationTurn]:
        if limit <= 0:
            return []
        async with self._session_factory() as session:
            assistants = list(
                await session.scalars(
                    select(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.conversation_id == conversation_id,
                        ConversationMessageRow.role == "assistant",
                    )
                    .order_by(
                        ConversationMessageRow.occurred_at.desc(),
                        ConversationMessageRow.id.desc(),
                    )
                    .limit(limit)
                )
            )
            event_keys = [
                row.inbound_event_key
                for row in assistants
                if row.inbound_event_key is not None
            ]
            customers = (
                {}
                if not event_keys
                else {
                    row.event_key: row
                    for row in await session.scalars(
                        select(ConversationMessageRow).where(
                            ConversationMessageRow.event_key.in_(event_keys)
                        )
                    )
                    if row.event_key is not None
                }
            )
        turns = [
            ConversationTurn(
                customer_text=customers[row.inbound_event_key].raw_text,
                assistant_text=row.raw_text,
                event_key=row.inbound_event_key,
                dedup_key=row.outbox_dedup_key,
            )
            for row in reversed(assistants)
            if row.inbound_event_key in customers
            and row.inbound_event_key is not None
            and row.outbox_dedup_key is not None
        ]
        return turns[-limit:]

    async def get_summary(
        self, conversation_id: str
    ) -> ConversationSummary | None:
        async with self._session_factory() as session:
            row = await session.get(ConversationSummaryRow, conversation_id)
        if row is None:
            return None
        return ConversationSummary(
            version=row.version,
            covered_until=_ensure_utc(row.covered_until),
            data=row.data,
            covered_message_id=row.covered_message_id,
        )

    async def _get_unsummarized_turns(
        self,
        conversation_id: str,
        *,
        covered_until: datetime | None,
        covered_message_id: int | None,
        limit: int,
    ) -> list[tuple[ConversationTurn, datetime, int]]:
        customer = aliased(ConversationMessageRow)
        assistant = aliased(ConversationMessageRow)
        statement = (
            select(
                customer.raw_text,
                assistant.raw_text,
                customer.event_key,
                assistant.outbox_dedup_key,
                customer.occurred_at,
                customer.id,
            )
            .join(customer, assistant.inbound_event_key == customer.event_key)
            .where(
                assistant.conversation_id == conversation_id,
                assistant.role == "assistant",
                customer.conversation_id == conversation_id,
                customer.role == "customer",
            )
            .order_by(customer.occurred_at, customer.id, assistant.id)
            .limit(limit)
        )
        if covered_until is not None:
            after_cursor = customer.occurred_at > covered_until
            after_cursor = or_(
                after_cursor,
                and_(
                    customer.occurred_at == covered_until,
                    customer.id > (covered_message_id or 0),
                ),
            )
            statement = statement.where(after_cursor)
        async with self._session_factory() as session:
            rows = list(await session.execute(statement))
        return [
            (
                ConversationTurn(
                    customer_text=customer_text,
                    assistant_text=assistant_text,
                    event_key=event_key,
                    dedup_key=dedup_key,
                ),
                _ensure_utc(occurred_at),
                message_id,
            )
            for (
                customer_text,
                assistant_text,
                event_key,
                dedup_key,
                occurred_at,
                message_id,
            ) in rows
            if event_key is not None and dedup_key is not None
        ]

    async def save_summary(
        self, conversation_id: str, summary: ConversationSummary
    ) -> None:
        values = {
            "conversation_id": conversation_id,
            "version": summary.version,
            "covered_until": summary.covered_until,
            "covered_message_id": summary.covered_message_id,
            "data": summary.data,
        }
        async with self._session_factory() as session:
            if self._engine.dialect.name == "postgresql":
                statement = postgres_insert(ConversationSummaryRow).values(**values)
            else:
                statement = sqlite_insert(ConversationSummaryRow).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["conversation_id"],
                set_={
                    "version": summary.version,
                    "covered_until": summary.covered_until,
                    "covered_message_id": summary.covered_message_id,
                    "data": summary.data,
                },
                where=and_(
                    ConversationSummaryRow.version < summary.version,
                    _summary_cursor_advances(summary),
                ),
            )
            await session.execute(statement)
            await session.commit()

    async def raw_messages(
        self, conversation_id: str
    ) -> list[dict[str, object]]:
        async with self._session_factory() as session:
            rows = list(
                await session.scalars(
                    select(ConversationMessageRow)
                    .where(
                        ConversationMessageRow.conversation_id == conversation_id
                    )
                    .order_by(ConversationMessageRow.id)
                )
            )
        return [
            {
                "role": row.role,
                "text": row.raw_text,
                "event_key": row.event_key or row.inbound_event_key,
                "outbox_dedup_key": row.outbox_dedup_key,
            }
            for row in rows
        ]

    async def maybe_summarize(self, conversation_id: str, trace_id: str) -> None:
        existing = await self.get_summary(conversation_id)
        turn_rows = await self._get_unsummarized_turns(
            conversation_id,
            covered_until=None if existing is None else existing.covered_until,
            covered_message_id=(
                None if existing is None else existing.covered_message_id
            ),
            limit=self._summary_policy.summarize_at_turns,
        )
        if not turn_rows:
            return
        turns = [turn for turn, _, _ in turn_rows]
        candidate = self._summary_policy.maybe_summarize(
            conversation_id,
            turns,
            existing,
            trace_id,
        )
        if candidate is None or candidate is existing:
            return
        covered_index = len(turns) - self._summary_policy.max_recent_turns - 1
        if covered_index < 0:
            return
        _, covered_at, covered_message_id = turn_rows[covered_index]
        if existing is not None and not _cursor_is_after(
            covered_at,
            covered_message_id,
            existing.covered_until,
            existing.covered_message_id,
        ):
            return
        await self.save_summary(
            conversation_id,
            replace(
                candidate,
                covered_until=covered_at,
                covered_message_id=covered_message_id,
            ),
        )


class PostgresConversationOutboxTransaction:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
        *,
        memory: PostgresConversationMemory | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self._memory = memory

    async def commit_event(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        command: OutboundCommand | None,
        handoff_request: HandoffRequest | None = None,
    ) -> CommitResult:
        async with self._session_factory() as session:
            result = CommitResult(
                status="committed", handoff_request=handoff_request
            )
            try:
                inbox_row = await session.scalar(
                    _for_update(
                        select(InboxRow).where(InboxRow.event_key == event.event_key),
                        self._engine,
                    )
                )
                if inbox_row is not None and inbox_row.processed_at is not None:
                    pending_handoff = await _pending_handoff(
                        session, event.event_key
                    )
                    await session.rollback()
                    return CommitResult(
                        status="duplicate",
                        reason="event_already_processed",
                        handoff_request=pending_handoff,
                    )
                if inbox_row is None:
                    inbox_row = InboxRow(
                        event_key=event.event_key,
                        event_payload=event_to_mapping(event),
                        received_at=event.received_at,
                    )
                    session.add(inbox_row)
                else:
                    inbox_row.event_payload = event_to_mapping(event)

                conversation_row = await session.scalar(
                    _for_update(
                        select(ConversationRow).where(
                            ConversationRow.conversation_id == event.conversation_id
                        ),
                        self._engine,
                    )
                )
                current_version = (
                    0 if conversation_row is None else conversation_row.version
                )
                if current_version != expected_version:
                    await session.rollback()
                    return CommitResult(
                        status="stale", reason="conversation_version_mismatch"
                    )

                if conversation.automation_state is AutomationState.HUMAN_ACTIVE:
                    command = None

                await session.execute(
                    _insert_ignore(
                        self._engine,
                        ConversationMessageRow,
                        conflict_columns=["event_key"],
                        values={
                            "conversation_id": event.conversation_id,
                            "role": "customer",
                            "raw_text": event.text if isinstance(event.text, str) else "",
                            "occurred_at": event.occurred_at,
                            "event_key": event.event_key,
                            "inbound_event_key": None,
                            "outbox_dedup_key": None,
                        },
                    )
                )

                if command is not None:
                    insert_result = await session.execute(
                        _insert_outbox_ignore(self._engine, command)
                    )
                    if insert_result.rowcount != 1:
                        result = CommitResult(
                            status="duplicate",
                            reason="dedup_key_exists",
                            handoff_request=handoff_request,
                        )
                    else:
                        await session.execute(
                            _insert_ignore(
                                self._engine,
                                ConversationMessageRow,
                                conflict_columns=["outbox_dedup_key"],
                                values={
                                    "conversation_id": event.conversation_id,
                                    "role": "assistant",
                                    "raw_text": _command_text(command),
                                    "occurred_at": _utc_now(),
                                    "event_key": None,
                                    "inbound_event_key": event.event_key,
                                    "outbox_dedup_key": command.dedup_key,
                                },
                            )
                        )

                if handoff_request is not None:
                    await session.execute(
                        _insert_handoff_ignore(
                            self._engine,
                            event.event_key,
                            handoff_request,
                        )
                    )

                if conversation_row is None:
                    insert_result = await session.execute(
                        _insert_conversation_ignore(self._engine, conversation)
                    )
                    if insert_result.rowcount != 1:
                        await session.rollback()
                        return CommitResult(
                            status="stale",
                            reason="conversation_version_mismatch",
                        )
                else:
                    conversation_row.payload = conversation_to_mapping(conversation)
                    conversation_row.version = conversation.version
                inbox_row.queued_at = None
                inbox_row.processed_at = _utc_now()

                await session.commit()
            except Exception:
                await session.rollback()
                raise
        if self._memory is not None:
            try:
                await self._memory.maybe_summarize(
                    event.conversation_id,
                    command.trace_id if command is not None else event.event_key,
                )
            except Exception:
                logger.warning(
                    "conversation summary failed",
                    extra={
                        "conversation_id": event.conversation_id,
                        "trace_id": (
                            command.trace_id if command is not None else event.event_key
                        ),
                    },
                )
        return result

    async def mark_handoff_complete(self, request_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(HandoffIntentRow)
                .where(HandoffIntentRow.request_id == request_id)
                .where(HandoffIntentRow.completed_at.is_(None))
                .values(completed_at=_utc_now())
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + database_url.removeprefix("postgresql://")
    if database_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + database_url.removeprefix("postgres://")
    return database_url


def _command_text(command: OutboundCommand) -> str:
    if isinstance(command.text, str) and command.text.strip():
        return command.text.strip()
    reply = command.template_variables.get("reply")
    return reply.strip() if isinstance(reply, str) else ""


def _engine_options(database_url: str) -> dict[str, Any]:
    if database_url == "sqlite+aiosqlite:///:memory:":
        return {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    return {}


def _conversation_row(conversation: Conversation) -> ConversationRow:
    return ConversationRow(
        conversation_id=conversation.conversation_id,
        payload=conversation_to_mapping(conversation),
        version=conversation.version,
    )


def _insert_conversation_ignore(
    engine: AsyncEngine, conversation: Conversation
):
    return _insert_ignore(
        engine,
        ConversationRow,
        conflict_columns=["conversation_id"],
        values={
            "conversation_id": conversation.conversation_id,
            "payload": conversation_to_mapping(conversation),
            "version": conversation.version,
        },
    )


def _insert_handoff_ignore(
    engine: AsyncEngine,
    event_key: str,
    request: HandoffRequest,
):
    return _insert_ignore(
        engine,
        HandoffIntentRow,
        conflict_columns=["event_key"],
        values={
            "request_id": request.request_id,
            "event_key": event_key,
            "request_payload": _handoff_request_to_mapping(request),
        },
    )


async def _pending_handoff(
    session: AsyncSession, event_key: str
) -> HandoffRequest | None:
    row = await session.scalar(
        select(HandoffIntentRow)
        .where(HandoffIntentRow.event_key == event_key)
        .where(HandoffIntentRow.completed_at.is_(None))
    )
    if row is None:
        return None
    return _handoff_request_from_mapping(row.request_payload)


def _handoff_request_to_mapping(request: HandoffRequest) -> dict[str, object]:
    return {
        "schema_version": 1,
        "request_id": request.request_id,
        "conversation_id": request.conversation_id,
        "reason": request.reason,
        "requested_by": request.requested_by,
        "requested_at": request.requested_at.isoformat(),
        "note": request.note,
    }


def _handoff_request_from_mapping(payload: dict[str, Any]) -> HandoffRequest:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported handoff intent schema_version")
    requested_at = datetime.fromisoformat(str(payload["requested_at"]))
    return HandoffRequest(
        request_id=str(payload["request_id"]),
        conversation_id=str(payload["conversation_id"]),
        reason=str(payload["reason"]),
        requested_by=str(payload["requested_by"]),
        requested_at=_ensure_utc(requested_at),
        note=None if payload.get("note") is None else str(payload["note"]),
    )


def _outbox_row_to_record(row: OutboxRow) -> OutboxRecord:
    return outbox_record_from_mapping(
        {
            "schema_version": 1,
            "command": row.command_payload,
            "state": row.state,
            "attempts": row.attempts,
            "provider_message_id": row.provider_message_id,
            "claimed_at": _optional_datetime_to_string(row.claimed_at),
            "lease_until": _optional_datetime_to_string(row.lease_until),
            "last_error": row.last_error,
        }
    )


def _insert_outbox_ignore(engine: AsyncEngine, command: OutboundCommand):
    return _insert_ignore(
        engine,
        OutboxRow,
        conflict_columns=["dedup_key"],
        values={
            "dedup_key": command.dedup_key,
            "conversation_id": command.conversation_id,
            "command_payload": command_to_mapping(command),
            "state": OutboxState.PENDING.value,
            "attempts": 0,
        },
    )


def _insert_ignore(
    engine: AsyncEngine,
    row_type: type[InboxRow]
    | type[ConversationRow]
    | type[OutboxRow]
    | type[HandoffIntentRow],
    *,
    conflict_columns: list[str],
    values: dict[str, Any],
):
    table = row_type.__table__
    dialect_name = engine.dialect.name
    if dialect_name == "postgresql":
        return postgres_insert(table).values(**values).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    if dialect_name == "sqlite":
        return sqlite_insert(table).values(**values).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    return table.insert().values(**values)


def _for_update(statement, engine: AsyncEngine, *, skip_locked: bool = False):
    if engine.dialect.name == "postgresql":
        return statement.with_for_update(skip_locked=skip_locked)
    return statement


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _effective_lease_seconds(lease_seconds: int | None) -> int:
    if lease_seconds is None:
        return DEFAULT_OUTBOX_LEASE_SECONDS
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return lease_seconds


def _validate_max_attempts(max_attempts: int | None) -> None:
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")


def _optional_datetime_to_string(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).isoformat()


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _cursor_is_after(
    occurred_at: datetime,
    message_id: int,
    covered_until: datetime,
    covered_message_id: int | None,
) -> bool:
    if occurred_at != covered_until:
        return occurred_at > covered_until
    return message_id > (covered_message_id or 0)


def _summary_cursor_advances(summary: ConversationSummary):
    later_timestamp = ConversationSummaryRow.covered_until < summary.covered_until
    if summary.covered_message_id is None:
        return later_timestamp
    return or_(
        later_timestamp,
        and_(
            ConversationSummaryRow.covered_until == summary.covered_until,
            or_(
                ConversationSummaryRow.covered_message_id.is_(None),
                ConversationSummaryRow.covered_message_id
                < summary.covered_message_id,
            ),
        ),
    )
