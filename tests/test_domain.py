from datetime import UTC, datetime, timedelta, timezone

import pytest

from whatsapp_support.domain import (
    AutomationState,
    Conversation,
    EventType,
    IdentityState,
    IdentityResult,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    OutboxState,
    WindowMode,
    window_mode,
)


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
    occurred_at: datetime,
    received_at: datetime | None = None,
    event_type: EventType = EventType.MESSAGE,
    event_key: str = "evt-1",
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conv-1",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id="wamid-1",
        event_type=event_type,
        occurred_at=occurred_at,
        received_at=received_at or occurred_at,
        payload_digest="digest-1",
        text="hello",
    )


def test_message_opens_24_hour_window_and_increments_version() -> None:
    start = utc_datetime(2026, 8, 22, 9, 0, 0)
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
    )

    updated, changed = conversation.apply_inbound(make_event(occurred_at=start))

    assert changed is True
    assert updated.customer_last_inbound_at == start
    assert updated.window_expires_at == start + timedelta(hours=24)
    assert updated.version == 1


def test_older_message_does_not_move_window_backwards() -> None:
    newer = utc_datetime(2026, 8, 22, 10, 0, 0)
    older = utc_datetime(2026, 8, 22, 9, 0, 0)
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        customer_last_inbound_at=newer,
        window_expires_at=newer + timedelta(hours=24),
        version=3,
    )

    updated, changed = conversation.apply_inbound(
        make_event(occurred_at=older, event_key="evt-2")
    )

    assert changed is False
    assert updated == conversation


def test_non_message_events_do_not_open_or_move_window() -> None:
    base = utc_datetime(2026, 8, 22, 11, 0, 0)
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        customer_last_inbound_at=base,
        window_expires_at=base + timedelta(hours=24),
        version=1,
    )

    updated, changed = conversation.apply_inbound(
        make_event(
            occurred_at=base + timedelta(minutes=10),
            event_type=EventType.STATUS,
            event_key="evt-3",
        )
    )

    assert changed is False
    assert updated == conversation


def test_window_mode_blocks_bot_output_during_human_handoff() -> None:
    now = utc_datetime(2026, 8, 22, 12, 0, 0)
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        automation_state=AutomationState.HUMAN_ACTIVE,
        window_expires_at=now + timedelta(hours=1),
    )

    assert window_mode(conversation, now) is WindowMode.BLOCKED


def test_window_mode_selects_template_after_expiry() -> None:
    now = utc_datetime(2026, 8, 22, 12, 0, 0)
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        window_expires_at=now,
    )

    assert window_mode(conversation, now) is WindowMode.TEMPLATE


def test_naive_timestamps_are_rejected() -> None:
    naive = datetime(2026, 8, 22, 9, 0, 0)

    with pytest.raises(ValueError):
        make_event(occurred_at=naive, received_at=naive)


def test_non_utc_aware_timestamps_are_rejected() -> None:
    utc_plus_8 = timezone(timedelta(hours=8))
    offset_time = datetime(2026, 8, 22, 9, 0, 0, tzinfo=utc_plus_8)

    with pytest.raises(ValueError):
        make_event(occurred_at=offset_time, received_at=offset_time)


def test_identity_result_denies_unverified_protected_scopes() -> None:
    result = IdentityResult(
        verified=False,
        scopes=frozenset({"customer:profile:read"}),
        subject_id="contact-1",
        status="pending",
        evidence=(),
    )

    assert result.allows("customer:profile:read") is False


def test_identity_result_rejects_non_utc_expiry() -> None:
    utc_plus_8 = timezone(timedelta(hours=8))

    with pytest.raises(ValueError, match="expires_at"):
        IdentityResult(
            verified=True,
            scopes=frozenset({"order:read"}),
            subject_id="customer-1",
            status="verified",
            expires_at=datetime(2026, 8, 22, 13, tzinfo=utc_plus_8),
        )


def test_conversation_keeps_identity_scope_and_expiry() -> None:
    conversation = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        identity_state=IdentityState.VERIFIED,
        identity_subject_id="subject-1",
        identity_scopes=frozenset({"order:read"}),
        identity_expires_at=utc_datetime(2026, 8, 22, 12),
    )

    assert conversation.identity_scopes == frozenset({"order:read"})
    assert conversation.identity_expires_at == utc_datetime(2026, 8, 22, 12)


def test_conversation_rejects_non_utc_identity_expiry() -> None:
    utc_plus_8 = timezone(timedelta(hours=8))

    with pytest.raises(ValueError):
        Conversation(
            conversation_id="conv-1",
            contact_id="contact-1",
            phone_number_id="phone-1",
            identity_expires_at=datetime(2026, 8, 22, 12, 0, 0, tzinfo=utc_plus_8),
        )


def test_outbox_states_are_stable_serializable_strings() -> None:
    assert [state.value for state in OutboxState] == [
        "PENDING",
        "SENDING",
        "RETRYABLE_FAILED",
        "SENT",
        "DEAD_LETTER",
    ]


def test_recipient_id_is_required_and_non_blank() -> None:
    timestamp = utc_datetime(2026, 8, 23)

    with pytest.raises(ValueError, match="recipient_id"):
        InboundEvent(
            event_key="evt-1",
            phone_number_id="phone-1",
            conversation_id="phone-1:wa-1",
            contact_id="wa-1",
            recipient_id=" ",
            provider_message_id="wamid-1",
            event_type=EventType.MESSAGE,
            occurred_at=timestamp,
            received_at=timestamp,
            payload_digest="digest-1",
            text="hello",
        )

    with pytest.raises(ValueError, match="recipient_id"):
        OutboundCommand(
            dedup_key="cmd-1",
            conversation_id="phone-1:wa-1",
            recipient_id="",
            mode=OutboundMode.FREEFORM,
            text="hello",
            template_name=None,
            template_locale=None,
            template_variables={},
            policy_decision="allow",
            trace_id="trace-1",
        )
