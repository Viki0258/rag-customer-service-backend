from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import (
    AutomationState,
    Conversation,
    EventType,
    IdentityProof,
    IdentityState,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    OutboxRecord,
    OutboxState,
)
from whatsapp_support.serialization import (
    command_from_mapping,
    command_to_mapping,
    conversation_from_mapping,
    conversation_to_mapping,
    event_from_mapping,
    event_to_mapping,
    outbox_record_from_mapping,
    outbox_record_to_mapping,
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
    event_type: EventType = EventType.MESSAGE,
    provider_message_id: str | None = "wamid-1",
    text: str | None = "hello",
    identity_proof: IdentityProof | None = None,
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    occurred_at = utc_datetime(2026, 8, 22, 9, 0, 0)
    received_at = utc_datetime(2026, 8, 22, 9, 0, 5)
    return InboundEvent(
        event_key=f"evt-{event_type.value}",
        phone_number_id="phone-1",
        conversation_id="conv-1",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=provider_message_id,
        event_type=event_type,
        occurred_at=occurred_at,
        received_at=received_at,
        payload_digest="digest-1",
        text=text,
        identity_proof=identity_proof,
    )


def test_event_mapping_round_trips_all_event_types_and_optional_fields() -> None:
    cases = [
        make_event(identity_proof=IdentityProof(kind="otp", value="123456")),
        make_event(event_type=EventType.STATUS, provider_message_id=None, text=None),
        make_event(event_type=EventType.ERROR, provider_message_id=None, text=None),
        make_event(
            event_type=EventType.UNSUPPORTED,
            provider_message_id=None,
            text=None,
        ),
    ]

    restored = [event_from_mapping(event_to_mapping(event)) for event in cases]

    assert restored == cases


def test_conversation_mapping_round_trips_identity_scope_order_and_utc_fields() -> None:
    original = Conversation(
        conversation_id="conv-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        customer_last_inbound_at=utc_datetime(2026, 8, 22, 9, 0, 0),
        window_expires_at=utc_datetime(2026, 8, 23, 9, 0, 0),
        automation_state=AutomationState.HUMAN_ACTIVE,
        identity_state=IdentityState.VERIFIED,
        identity_subject_id="subject-1",
        identity_scopes=frozenset({"orders:read", "profile:read"}),
        identity_expires_at=utc_datetime(2026, 8, 22, 12, 0, 0),
        version=4,
    )

    payload = conversation_to_mapping(original)
    restored = conversation_from_mapping(payload)

    assert payload["identity_scopes"] == ["orders:read", "profile:read"]
    assert restored == original


def test_command_mapping_round_trips_freeform_and_template_modes() -> None:
    cases = [
        OutboundCommand(
            dedup_key="cmd-freeform",
            conversation_id="conv-1",
            recipient_id="recipient-1",
            mode=OutboundMode.FREEFORM,
            text="hello",
            template_name=None,
            template_locale=None,
            template_variables={},
            policy_decision="allow",
            trace_id="trace-freeform",
        ),
        OutboundCommand(
            dedup_key="cmd-template",
            conversation_id="conv-1",
            recipient_id="recipient-1",
            mode=OutboundMode.TEMPLATE,
            text=None,
            template_name="order_update",
            template_locale="en_US",
            template_variables={"order_id": "123", "status": "shipped"},
            policy_decision="allow_template",
            trace_id="trace-template",
        ),
    ]

    restored = [command_from_mapping(command_to_mapping(command)) for command in cases]

    assert restored == cases


def test_outbox_record_mapping_round_trips_state_and_utc_lease_fields() -> None:
    command = OutboundCommand(
        dedup_key="cmd-record",
        conversation_id="conv-1",
        recipient_id="recipient-1",
        mode=OutboundMode.FREEFORM,
        text="hello",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id="trace-record",
    )
    original = OutboxRecord(
        command=command,
        state=OutboxState.SENDING,
        attempts=2,
        provider_message_id="wamid-1",
        claimed_at=utc_datetime(2026, 8, 22, 9, 1, 0),
        lease_until=utc_datetime(2026, 8, 22, 9, 2, 0),
        last_error="timeout",
    )

    restored = outbox_record_from_mapping(outbox_record_to_mapping(original))

    assert restored == original


def test_event_from_mapping_rejects_malformed_payloads() -> None:
    payload = event_to_mapping(
        make_event(identity_proof=IdentityProof(kind="otp", value="123456"))
    )

    with pytest.raises(ValueError, match="schema_version"):
        event_from_mapping({**payload, "schema_version": 99})

    with pytest.raises(ValueError, match="occurred_at"):
        event_from_mapping({**payload, "occurred_at": "2026-08-22T09:00:00"})

    with pytest.raises(ValueError, match="identity_proof"):
        event_from_mapping({**payload, "identity_proof": {"kind": "otp"}})

    malformed_without_identity_proof = dict(payload)
    malformed_without_identity_proof.pop("identity_proof")

    with pytest.raises(ValueError, match="identity_proof"):
        event_from_mapping(malformed_without_identity_proof)

    malformed_without_recipient_id = dict(payload)
    malformed_without_recipient_id.pop("recipient_id")

    with pytest.raises(ValueError, match="recipient_id"):
        event_from_mapping(malformed_without_recipient_id)


def test_conversation_from_mapping_rejects_unsorted_or_non_string_scopes() -> None:
    payload = conversation_to_mapping(
        Conversation(
            conversation_id="conv-1",
            contact_id="contact-1",
            phone_number_id="phone-1",
            identity_scopes=frozenset({"a", "b"}),
        )
    )

    with pytest.raises(ValueError, match="identity_scopes"):
        conversation_from_mapping({**payload, "identity_scopes": ["b", "a"]})

    with pytest.raises(ValueError, match="identity_scopes"):
        conversation_from_mapping({**payload, "identity_scopes": ["a", 3]})


@pytest.mark.parametrize("automation_state", list(AutomationState))
@pytest.mark.parametrize("identity_state", list(IdentityState))
def test_conversation_mapping_round_trips_all_automation_and_identity_states(
    automation_state: AutomationState,
    identity_state: IdentityState,
) -> None:
    original = Conversation(
        conversation_id=f"conv-{automation_state.value}-{identity_state.value}",
        contact_id="contact-1",
        phone_number_id="phone-1",
        automation_state=automation_state,
        identity_state=identity_state,
    )

    restored = conversation_from_mapping(conversation_to_mapping(original))

    assert restored == original


def test_command_from_mapping_rejects_invalid_template_variables_shape() -> None:
    payload = command_to_mapping(
        OutboundCommand(
            dedup_key="cmd-template",
            conversation_id="conv-1",
            recipient_id="recipient-1",
            mode=OutboundMode.TEMPLATE,
            text=None,
            template_name="order_update",
            template_locale="en_US",
            template_variables={"order_id": "123"},
            policy_decision="allow_template",
            trace_id="trace-template",
        )
    )

    with pytest.raises(ValueError, match="template_variables"):
        command_from_mapping({**payload, "template_variables": {"order_id": 123}})


def test_command_mapping_preserves_recipient_id_and_requires_the_key() -> None:
    command = OutboundCommand(
        dedup_key="cmd-recipient",
        conversation_id="phone-1:wa-1",
        recipient_id="wa-15551234567",
        mode=OutboundMode.FREEFORM,
        text="hello",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id="trace-recipient",
    )
    payload = command_to_mapping(command)

    assert payload["recipient_id"] == "wa-15551234567"
    assert command_from_mapping(payload) == command
    with pytest.raises(ValueError, match="recipient_id"):
        command_from_mapping(
            {key: value for key, value in payload.items() if key != "recipient_id"}
        )
