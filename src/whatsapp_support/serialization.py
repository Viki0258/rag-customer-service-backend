from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

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


SCHEMA_VERSION = 1


def event_to_mapping(event: InboundEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_key": event.event_key,
        "phone_number_id": event.phone_number_id,
        "conversation_id": event.conversation_id,
        "contact_id": event.contact_id,
        "recipient_id": event.recipient_id,
        "provider_message_id": event.provider_message_id,
        "event_type": event.event_type.value,
        "occurred_at": _datetime_to_string(event.occurred_at),
        "received_at": _datetime_to_string(event.received_at),
        "payload_digest": event.payload_digest,
        "text": event.text,
        "identity_proof": None,
    }
    if event.identity_proof is not None:
        payload["identity_proof"] = {
            "kind": event.identity_proof.kind,
            "value": event.identity_proof.value,
        }
    return payload


def event_from_mapping(payload: object) -> InboundEvent:
    mapping = _require_mapping(payload, "payload")
    _require_schema_version(mapping)
    identity_payload = _require_nullable_field(mapping, "identity_proof")
    identity_proof: IdentityProof | None = None
    if identity_payload is not None:
        identity_mapping = _require_mapping(identity_payload, "identity_proof")
        identity_proof = IdentityProof(
            kind=_require_string(identity_mapping, "kind", container_name="identity_proof"),
            value=_require_string(identity_mapping, "value", container_name="identity_proof"),
        )
    return InboundEvent(
        event_key=_require_string(mapping, "event_key"),
        phone_number_id=_require_string(mapping, "phone_number_id"),
        conversation_id=_require_string(mapping, "conversation_id"),
        contact_id=_require_string(mapping, "contact_id"),
        recipient_id=_require_string(mapping, "recipient_id"),
        provider_message_id=_require_optional_string(mapping, "provider_message_id"),
        event_type=_require_enum(mapping, "event_type", EventType),
        occurred_at=_require_datetime(mapping, "occurred_at"),
        received_at=_require_datetime(mapping, "received_at"),
        payload_digest=_require_string(mapping, "payload_digest"),
        text=_require_optional_string(mapping, "text"),
        identity_proof=identity_proof,
    )


def conversation_to_mapping(conversation: Conversation) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": conversation.conversation_id,
        "contact_id": conversation.contact_id,
        "phone_number_id": conversation.phone_number_id,
        "customer_last_inbound_at": _optional_datetime_to_string(
            conversation.customer_last_inbound_at
        ),
        "window_expires_at": _optional_datetime_to_string(
            conversation.window_expires_at
        ),
        "automation_state": conversation.automation_state.value,
        "identity_state": conversation.identity_state.value,
        "identity_subject_id": conversation.identity_subject_id,
        "identity_scopes": sorted(conversation.identity_scopes),
        "identity_expires_at": _optional_datetime_to_string(
            conversation.identity_expires_at
        ),
        "version": conversation.version,
    }


def conversation_from_mapping(payload: object) -> Conversation:
    mapping = _require_mapping(payload, "payload")
    _require_schema_version(mapping)
    scopes = _require_sorted_string_list(mapping, "identity_scopes")
    return Conversation(
        conversation_id=_require_string(mapping, "conversation_id"),
        contact_id=_require_string(mapping, "contact_id"),
        phone_number_id=_require_string(mapping, "phone_number_id"),
        customer_last_inbound_at=_require_optional_datetime(
            mapping, "customer_last_inbound_at"
        ),
        window_expires_at=_require_optional_datetime(mapping, "window_expires_at"),
        automation_state=_require_enum(mapping, "automation_state", AutomationState),
        identity_state=_require_enum(mapping, "identity_state", IdentityState),
        identity_subject_id=_require_optional_string(mapping, "identity_subject_id"),
        identity_scopes=frozenset(scopes),
        identity_expires_at=_require_optional_datetime(mapping, "identity_expires_at"),
        version=_require_int(mapping, "version"),
    )


def command_to_mapping(command: OutboundCommand) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dedup_key": command.dedup_key,
        "conversation_id": command.conversation_id,
        "recipient_id": command.recipient_id,
        "mode": command.mode.value,
        "text": command.text,
        "template_name": command.template_name,
        "template_locale": command.template_locale,
        "template_variables": dict(command.template_variables),
        "policy_decision": command.policy_decision,
        "trace_id": command.trace_id,
        "retrieved_chunk_ids": list(command.retrieved_chunk_ids),
        "supported_chunk_ids": list(command.supported_chunk_ids),
        "prompt_version": command.prompt_version,
        "model_name": command.model_name,
    }


def command_from_mapping(payload: object) -> OutboundCommand:
    mapping = _require_mapping(payload, "payload")
    _require_schema_version(mapping)
    return OutboundCommand(
        dedup_key=_require_string(mapping, "dedup_key"),
        conversation_id=_require_string(mapping, "conversation_id"),
        recipient_id=_require_string(mapping, "recipient_id"),
        mode=_require_enum(mapping, "mode", OutboundMode),
        text=_require_optional_string(mapping, "text"),
        template_name=_require_optional_string(mapping, "template_name"),
        template_locale=_require_optional_string(mapping, "template_locale"),
        template_variables=_require_string_dict(mapping, "template_variables"),
        policy_decision=_require_string(mapping, "policy_decision"),
        trace_id=_require_string(mapping, "trace_id"),
        retrieved_chunk_ids=tuple(mapping.get("retrieved_chunk_ids", [])),
        supported_chunk_ids=tuple(mapping.get("supported_chunk_ids", [])),
        prompt_version=_require_optional_string(mapping, "prompt_version"),
        model_name=_require_optional_string(mapping, "model_name"),
    )


def outbox_record_to_mapping(record: OutboxRecord) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command_to_mapping(record.command),
        "state": record.state.value,
        "attempts": record.attempts,
        "provider_message_id": record.provider_message_id,
        "claimed_at": _optional_datetime_to_string(record.claimed_at),
        "lease_until": _optional_datetime_to_string(record.lease_until),
        "last_error": record.last_error,
    }


def outbox_record_from_mapping(payload: object) -> OutboxRecord:
    mapping = _require_mapping(payload, "payload")
    _require_schema_version(mapping)
    return OutboxRecord(
        command=command_from_mapping(
            _require_mapping(mapping.get("command"), "command")
        ),
        state=_require_enum(mapping, "state", OutboxState),
        attempts=_require_int(mapping, "attempts"),
        provider_message_id=_require_optional_string(mapping, "provider_message_id"),
        claimed_at=_require_optional_datetime(mapping, "claimed_at"),
        lease_until=_require_optional_datetime(mapping, "lease_until"),
        last_error=_require_optional_string(mapping, "last_error"),
    )


def _require_schema_version(mapping: dict[str, object]) -> None:
    schema_version = mapping.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
    return value


def _require_nullable_field(mapping: dict[str, object], field_name: str) -> object | None:
    if field_name not in mapping:
        raise ValueError(f"payload.{field_name} must be present")
    return mapping[field_name]


def _require_string(
    mapping: dict[str, object],
    field_name: str,
    *,
    container_name: str = "payload",
) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{container_name}.{field_name} must be a string")
    return value


def _require_optional_string(mapping: dict[str, object], field_name: str) -> str | None:
    value = mapping.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"payload.{field_name} must be a string or null")
    return value


def _require_int(mapping: dict[str, object], field_name: str) -> int:
    value = mapping.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"payload.{field_name} must be an integer")
    return value


def _require_enum(
    mapping: dict[str, object], field_name: str, enum_type: type[Enum]
) -> Enum:
    raw = _require_string(mapping, field_name)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise ValueError(f"payload.{field_name} must be a valid {enum_type.__name__}") from exc


def _datetime_to_string(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
        raise ValueError("datetime values must be UTC-aware")
    return value.isoformat()


def _optional_datetime_to_string(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _datetime_to_string(value)


def _require_datetime(mapping: dict[str, object], field_name: str) -> datetime:
    raw = _require_string(mapping, field_name)
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"payload.{field_name} must be an ISO-8601 datetime") from exc
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"payload.{field_name} must be a UTC datetime")
    return value.astimezone(UTC)


def _require_optional_datetime(
    mapping: dict[str, object], field_name: str
) -> datetime | None:
    raw = mapping.get(field_name)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"payload.{field_name} must be an ISO-8601 datetime or null")
    return _require_datetime(mapping, field_name)


def _require_sorted_string_list(
    mapping: dict[str, object], field_name: str
) -> list[str]:
    raw = mapping.get(field_name)
    if not isinstance(raw, list):
        raise ValueError(f"payload.{field_name} must be a list of strings")
    if any(not isinstance(item, str) for item in raw):
        raise ValueError(f"payload.{field_name} must contain only strings")
    if raw != sorted(raw):
        raise ValueError(f"payload.{field_name} must be sorted")
    return list(raw)


def _require_string_dict(mapping: dict[str, object], field_name: str) -> dict[str, str]:
    raw = mapping.get(field_name)
    nested = _require_mapping(raw, field_name)
    result: dict[str, str] = {}
    for key, value in nested.items():
        if not isinstance(value, str):
            raise ValueError(f"{field_name} values must be strings")
        result[key] = value
    return result
