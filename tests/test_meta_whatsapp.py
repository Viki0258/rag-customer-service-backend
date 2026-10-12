from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import (
    OutboundCommand,
    OutboundMode,
    OutboxState,
)
from whatsapp_support.fakes import InMemoryOutboxStore
from whatsapp_support.meta_whatsapp import (
    AsyncHttpTransport,
    HttpResponse,
    MetaProviderError,
    MetaWhatsAppAdapter,
)
from whatsapp_support.worker import OutboxDispatcher


SENT_AT = datetime(2026, 8, 23, 9, 30, tzinfo=UTC)
TOKEN = "secret-token"
PHONE_NUMBER = "15551234567"
LEAKED_BODY = {
    "error": {
        "message": f"token={TOKEN} phone={PHONE_NUMBER} provider-body-secret",
    }
}


@dataclass(frozen=True)
class RecordedCall:
    url: str
    headers: dict[str, str]
    json_body: dict[str, object]
    timeout: float


class RecordingTransport:
    def __init__(
        self,
        *responses: HttpResponse,
        errors: list[BaseException] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.errors = list(errors or [])
        self.calls: list[RecordedCall] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object],
        timeout: float,
    ) -> HttpResponse:
        self.calls.append(RecordedCall(url, headers, json_body, timeout))
        if self.errors:
            raise self.errors.pop(0)
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def make_command(
    *,
    recipient_id: str = PHONE_NUMBER,
    text: str | None = "hello",
    mode: OutboundMode = OutboundMode.FREEFORM,
    template_name: str | None = None,
    template_locale: str | None = None,
    template_variables: dict[str, str] | None = None,
    dedup_key: str = "command-1",
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conversation-1",
        recipient_id=recipient_id,
        mode=mode,
        text=text,
        template_name=template_name,
        template_locale=template_locale,
        template_variables=template_variables or {},
        policy_decision="allow",
        trace_id=f"trace-{dedup_key}",
    )


def make_template_command(
    *, template_variables: dict[str, str] | None = None
) -> OutboundCommand:
    return make_command(
        mode=OutboundMode.TEMPLATE,
        text=None,
        template_name="order_update",
        template_locale="en_US",
        template_variables=template_variables
        or {"order_id": "O-9", "status": "shipped"},
    )


def configured_adapter(
    transport: AsyncHttpTransport,
    *,
    timeout: float = 7.5,
) -> MetaWhatsAppAdapter:
    return MetaWhatsAppAdapter(
        TOKEN,
        "business-phone-1",
        "https://graph.example/v20.0",
        transport,
        clock=lambda: SENT_AT,
        timeout=timeout,
    )


@pytest.mark.asyncio
async def test_text_send_uses_recipient_id_and_returns_provider_id() -> None:
    transport = RecordingTransport(
        HttpResponse(200, {"messages": [{"id": "wamid-out-1"}]})
    )
    adapter = MetaWhatsAppAdapter(
        TOKEN,
        "business-phone-1",
        "https://graph.example/v20.0",
        transport,
        clock=lambda: SENT_AT,
    )

    result = await adapter.send(make_command(recipient_id=PHONE_NUMBER))

    assert result.provider_message_id == "wamid-out-1"
    assert result.accepted is True
    assert result.status == "accepted"
    assert result.sent_at == SENT_AT
    call = transport.calls[0]
    assert call.url == "https://graph.example/v20.0/business-phone-1/messages"
    assert call.headers == {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    assert call.json_body == {
        "messaging_product": "whatsapp",
        "to": PHONE_NUMBER,
        "type": "text",
        "text": {"body": "hello"},
    }
    assert call.timeout == 10.0


@pytest.mark.asyncio
async def test_success_response_uses_first_non_empty_provider_id() -> None:
    transport = RecordingTransport(
        HttpResponse(200, {"messages": [{"id": ""}, {"id": "wamid-out-2"}]})
    )
    adapter = configured_adapter(transport)

    result = await adapter.send(make_command())

    assert result.provider_message_id == "wamid-out-2"


@pytest.mark.asyncio
async def test_template_variables_keep_command_order_and_invalid_template_never_calls_transport() -> None:
    transport = RecordingTransport(
        HttpResponse(200, {"messages": [{"id": "wamid-out-2"}]})
    )
    adapter = configured_adapter(transport)
    command = make_template_command(
        template_variables={"order_id": "O-9", "status": "shipped"}
    )

    await adapter.send(command)
    assert transport.calls[0].json_body == {
        "messaging_product": "whatsapp",
        "to": PHONE_NUMBER,
        "type": "template",
        "template": {
            "name": "order_update",
            "language": {"code": "en_US"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": "O-9"},
                        {"type": "text", "text": "shipped"},
                    ],
                }
            ],
        },
    }

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(replace(command, template_name=None))
    assert excinfo.value.error_code == "invalid_command"
    assert excinfo.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template_variables",
    [None, [], "not-a-mapping", 42],
)
async def test_malformed_template_variable_shapes_are_safe_invalid_commands(
    template_variables: object,
) -> None:
    transport = RecordingTransport(
        HttpResponse(200, {"messages": [{"id": "should-not-send"}]})
    )
    adapter = configured_adapter(transport)
    invalid = replace(
        make_template_command(),
        template_variables=template_variables,  # type: ignore[arg-type]
    )

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(invalid)

    assert excinfo.value.error_code == "invalid_command"
    assert excinfo.value.retryable is False
    assert transport.calls == []


@pytest.mark.asyncio
async def test_invalid_template_variable_shape_never_calls_transport() -> None:
    transport = RecordingTransport(
        HttpResponse(200, {"messages": [{"id": "should-not-send"}]})
    )
    adapter = configured_adapter(transport)
    invalid = replace(
        make_template_command(),
        template_variables={"valid": "yes", 9: "not-a-string"},  # type: ignore[dict-item]
    )

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(invalid)

    assert excinfo.value.error_code == "invalid_command"
    assert transport.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code", "retryable"),
    [
        (HttpResponse(429, LEAKED_BODY), "rate_limited", True),
        (HttpResponse(500, LEAKED_BODY), "service_unavailable", True),
        (HttpResponse(502, LEAKED_BODY), "service_unavailable", True),
        (HttpResponse(503, LEAKED_BODY), "service_unavailable", True),
        (HttpResponse(400, LEAKED_BODY), "invalid_request", False),
        (HttpResponse(401, LEAKED_BODY), "authentication_failed", False),
        (HttpResponse(403, LEAKED_BODY), "authentication_failed", False),
    ],
)
async def test_http_statuses_map_to_safe_provider_errors(
    response: HttpResponse,
    error_code: str,
    retryable: bool,
) -> None:
    transport = RecordingTransport(response)
    adapter = configured_adapter(transport)

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(make_command())

    error = excinfo.value
    assert error.error_code == error_code
    assert error.retryable is retryable
    assert str(error) == error_code
    assert TOKEN not in str(error)
    assert PHONE_NUMBER not in str(error)
    assert "provider-body-secret" not in str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "error_code"),
    [
        (ConnectionError(f"network leaked {TOKEN} {PHONE_NUMBER}"), "network_error"),
        (TimeoutError(f"timeout leaked {TOKEN} {PHONE_NUMBER}"), "timeout"),
        (RuntimeError(f"unexpected leaked {TOKEN} {PHONE_NUMBER}"), "network_error"),
    ],
)
async def test_transport_failures_map_to_safe_provider_errors(
    transport_error: BaseException,
    error_code: str,
) -> None:
    transport = RecordingTransport(errors=[transport_error])
    adapter = configured_adapter(transport)

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(make_command())

    assert excinfo.value.error_code == error_code
    assert excinfo.value.retryable is True
    assert str(excinfo.value) == error_code
    assert TOKEN not in repr(excinfo.value)
    assert PHONE_NUMBER not in repr(excinfo.value)


@pytest.mark.asyncio
async def test_success_without_message_id_is_safe_retryable_protocol_error() -> None:
    transport = RecordingTransport(HttpResponse(200, {"messages": [{}]}))
    adapter = configured_adapter(transport)

    with pytest.raises(MetaProviderError) as excinfo:
        await adapter.send(make_command())

    assert excinfo.value.error_code == "protocol_error"
    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_adapter_aclose_closes_injected_transport() -> None:
    transport = RecordingTransport()
    adapter = configured_adapter(transport)

    await adapter.aclose()

    assert transport.closed is True


@pytest.mark.asyncio
async def test_retryable_meta_failure_sets_retryable_failed_before_dead_letter() -> None:
    transport = RecordingTransport(
        HttpResponse(503, LEAKED_BODY),
        HttpResponse(503, LEAKED_BODY),
    )
    adapter = configured_adapter(transport)
    outbox = InMemoryOutboxStore()
    command = make_command(dedup_key="retry-meta")
    await outbox.enqueue(command)
    dispatcher = OutboxDispatcher(outbox, adapter, max_attempts=2)

    first = await dispatcher.dispatch_once()
    record_after_first = (await outbox.records())[0]
    second = await dispatcher.dispatch_once()
    record_after_second = (await outbox.records())[0]

    assert first.status == "retryable"
    assert first.error == "service_unavailable"
    assert record_after_first.state is OutboxState.RETRYABLE_FAILED
    assert second.status == "dead_letter"
    assert second.error == "service_unavailable"
    assert record_after_second.state is OutboxState.DEAD_LETTER
    assert record_after_second.last_error == "service_unavailable"
    assert TOKEN not in repr(first)
    assert PHONE_NUMBER not in repr(first)
    assert "provider-body-secret" not in repr(first)
    assert TOKEN not in (record_after_second.last_error or "")
    assert PHONE_NUMBER not in (record_after_second.last_error or "")


@pytest.mark.asyncio
async def test_permanent_meta_failure_dead_letters_without_retry() -> None:
    transport = RecordingTransport(HttpResponse(401, LEAKED_BODY))
    adapter = configured_adapter(transport)
    outbox = InMemoryOutboxStore()
    await outbox.enqueue(make_command(dedup_key="permanent-meta"))
    dispatcher = OutboxDispatcher(outbox, adapter)

    result = await dispatcher.dispatch_once()
    record = (await outbox.records())[0]

    assert result.status == "dead_letter"
    assert result.error == "authentication_failed"
    assert record.state is OutboxState.DEAD_LETTER
    assert record.last_error == "authentication_failed"
