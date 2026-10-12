from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

from whatsapp_support.domain import OutboundCommand, OutboundMode, ProviderSendResult


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    payload: object


class AsyncHttpTransport(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
        timeout: float,
    ) -> HttpResponse: ...

    async def aclose(self) -> None: ...


class MetaProviderError(RuntimeError):
    def __init__(self, error_code: str, retryable: bool) -> None:
        self.error_code = error_code
        self.retryable = retryable
        super().__init__(error_code)


class HttpxTransport:
    def __init__(self) -> None:
        import httpx

        self._client = httpx.AsyncClient()

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
        timeout: float,
    ) -> HttpResponse:
        import httpx

        try:
            response = await self._client.post(
                url,
                headers=dict(headers),
                json=dict(json_body),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError from None
        except httpx.NetworkError as exc:
            raise ConnectionError from None

        try:
            payload = response.json()
        except Exception:
            payload = {}
        return HttpResponse(response.status_code, payload)

    async def aclose(self) -> None:
        await self._client.aclose()


class MetaWhatsAppAdapter:
    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        graph_base_url: str,
        transport: AsyncHttpTransport,
        *,
        clock: Callable[[], datetime] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._graph_base_url = graph_base_url
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout = timeout

    async def send(self, command: OutboundCommand) -> ProviderSendResult:
        payload = _payload_for(command)
        url = (
            self._graph_base_url.rstrip("/")
            + "/"
            + self._phone_number_id
            + "/messages"
        )
        try:
            response = await self._transport.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json",
                },
                json_body=payload,
                timeout=self._timeout,
            )
        except TimeoutError:
            raise MetaProviderError("timeout", retryable=True) from None
        except (ConnectionError, OSError):
            raise MetaProviderError("network_error", retryable=True) from None
        except Exception:
            raise MetaProviderError("network_error", retryable=True) from None

        if response.status_code == 429:
            raise MetaProviderError("rate_limited", retryable=True)
        if 500 <= response.status_code <= 599:
            raise MetaProviderError("service_unavailable", retryable=True)
        if 400 <= response.status_code <= 499:
            if response.status_code in {401, 403}:
                raise MetaProviderError("authentication_failed", retryable=False)
            raise MetaProviderError("invalid_request", retryable=False)
        if not 200 <= response.status_code <= 299:
            raise MetaProviderError("protocol_error", retryable=True)

        provider_message_id = _provider_message_id(response.payload)
        if provider_message_id is None:
            raise MetaProviderError("protocol_error", retryable=True)
        return ProviderSendResult(
            provider_message_id=provider_message_id,
            accepted=True,
            sent_at=self._clock(),
            status="accepted",
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


def _payload_for(command: OutboundCommand) -> dict[str, object]:
    if not isinstance(command.recipient_id, str) or not command.recipient_id.strip():
        raise MetaProviderError("invalid_command", retryable=False)
    if command.mode is OutboundMode.FREEFORM:
        if not isinstance(command.text, str) or not command.text.strip():
            raise MetaProviderError("invalid_command", retryable=False)
        return {
            "messaging_product": "whatsapp",
            "to": command.recipient_id,
            "type": "text",
            "text": {"body": command.text},
        }
    if command.mode is OutboundMode.TEMPLATE:
        if not command.template_name or not command.template_locale:
            raise MetaProviderError("invalid_command", retryable=False)
        if not isinstance(command.template_variables, Mapping):
            raise MetaProviderError("invalid_command", retryable=False)
        parameters = [
            {"type": "text", "text": value}
            for key, value in command.template_variables.items()
            if isinstance(key, str) and isinstance(value, str)
        ]
        if len(parameters) != len(command.template_variables):
            raise MetaProviderError("invalid_command", retryable=False)
        return {
            "messaging_product": "whatsapp",
            "to": command.recipient_id,
            "type": "template",
            "template": {
                "name": command.template_name,
                "language": {"code": command.template_locale},
                "components": [{"type": "body", "parameters": parameters}],
            },
        }
    raise MetaProviderError("invalid_command", retryable=False)


def _provider_message_id(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        provider_message_id = message.get("id")
        if isinstance(provider_message_id, str) and provider_message_id.strip():
            return provider_message_id
    return None
