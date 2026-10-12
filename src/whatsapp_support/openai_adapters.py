from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from whatsapp_support.domain import ModelRequest, StructuredModelResult
from whatsapp_support.meta_whatsapp import AsyncHttpTransport, HttpResponse
from whatsapp_support.vector_validation import validate_vector


_STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reply": {"type": "string"},
        "supported_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "handoff": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["reply", "supported_evidence_ids", "handoff", "reason"],
}


class OpenAIAdapterError(RuntimeError):
    def __init__(self, error_code: str, retryable: bool) -> None:
        self.error_code = error_code
        self.retryable = retryable
        super().__init__(error_code)


class OpenAIEmbeddingAdapter:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        transport: AsyncHttpTransport,
        *,
        timeout_seconds: float,
        max_retries: int,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 10,
    ) -> None:
        self._base_url = validate_api_base_url(base_url)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        self._api_key = _require_text(api_key, "api_key")
        self.model_name = _require_text(model, "model")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self._transport = transport
        self._timeout_seconds = _require_positive_number(
            timeout_seconds, "timeout_seconds"
        )
        self._max_retries = _require_non_negative_int(max_retries, "max_retries")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self._batch_size:
            vectors = []
            for offset in range(0, len(texts), self._batch_size):
                vectors.extend(await self.embed(texts[offset:offset + self._batch_size]))
            return vectors
        payload = {
            "model": self.model_name,
            "input": list(texts),
            "dimensions": self.dimension,
        }
        response = await _post_with_retries(
            self._transport,
            f"{self._base_url}/embeddings",
            headers=_authorization_headers(self._api_key),
            json_body=payload,
            timeout_seconds=self._timeout_seconds,
            max_retries=self._max_retries,
            error_code="embedding_failed",
        )
        try:
            return _parse_embeddings_payload(
                response.payload,
                expected_count=len(texts),
                dimension=self.dimension,
            )
        except ValueError:
            raise OpenAIAdapterError("embedding_failed", retryable=False) from None

    async def aclose(self) -> None:
        await self._transport.aclose()


class OpenAIResponseAdapter:
    def __init__(
        self,
        api_key: str,
        model: str,
        transport: AsyncHttpTransport,
        *,
        timeout_seconds: float,
        max_retries: int,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._base_url = validate_api_base_url(base_url)
        self._api_key = _require_text(api_key, "api_key")
        self._model = _require_text(model, "model")
        self._transport = transport
        self._timeout_seconds = _require_positive_number(
            timeout_seconds, "timeout_seconds"
        )
        self._max_retries = _require_non_negative_int(max_retries, "max_retries")

    async def generate(self, request: ModelRequest) -> StructuredModelResult:
        payload = {
            "model": self._model,
            "input": request.inputs["messages"],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "rag_answer",
                    "schema": _STRUCTURED_OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        }
        response = await _post_with_retries(
            self._transport,
            f"{self._base_url}/responses",
            headers=_authorization_headers(self._api_key),
            json_body=payload,
            timeout_seconds=self._timeout_seconds,
            max_retries=self._max_retries,
            error_code="model_failed",
        )
        try:
            raw_text, content = _parse_structured_output(response.payload)
        except ValueError:
            raise OpenAIAdapterError("model_schema_invalid", retryable=False) from None
        tokens_in, tokens_out = _parse_usage(response.payload)
        return StructuredModelResult(
            trace_id=request.trace_id,
            model_name=self._model,
            prompt_version=request.prompt_version,
            content=content,
            raw_text=raw_text,
            created_at=datetime.now(UTC),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


async def _post_with_retries(
    transport: AsyncHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    json_body: Mapping[str, object],
    timeout_seconds: float,
    max_retries: int,
    error_code: str,
) -> HttpResponse:
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            response = await transport.post(
                url,
                headers=headers,
                json_body=json_body,
                timeout=timeout_seconds,
            )
        except TimeoutError:
            if attempt < max_retries:
                continue
            raise OpenAIAdapterError(error_code, retryable=True) from None
        except (ConnectionError, OSError):
            if attempt < max_retries:
                continue
            raise OpenAIAdapterError(error_code, retryable=True) from None

        if response.status_code in {429} or 500 <= response.status_code <= 599:
            if attempt < max_retries:
                continue
            raise OpenAIAdapterError(error_code, retryable=True)
        if 200 <= response.status_code <= 299:
            return response
        raise OpenAIAdapterError(error_code, retryable=False)
    raise OpenAIAdapterError(error_code, retryable=True)


def _parse_embeddings_payload(
    payload: object,
    *,
    expected_count: int,
    dimension: int,
) -> list[list[float]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("data must be a list")
    if len(rows) != expected_count:
        raise ValueError("embedding row count mismatch")
    indexed_vectors: dict[int, list[float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("embedding row must be an object")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("embedding index must be an integer")
        if index < 0 or index >= expected_count:
            raise ValueError("embedding index out of range")
        if index in indexed_vectors:
            raise ValueError("embedding index must be unique")
        indexed_vectors[index] = validate_vector(
            row.get("embedding"), dimension, label="embedding vector"
        )
    if set(indexed_vectors) != set(range(expected_count)):
        raise ValueError("embedding indexes must be contiguous")
    vectors: list[list[float]] = []
    for index in range(expected_count):
        vector = indexed_vectors[index]
        vectors.append(vector)
    return vectors


def _parse_structured_output(payload: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("output must be a list")
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, Mapping):
                continue
            if chunk.get("type") != "output_text":
                continue
            raw_text = chunk.get("text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("output_text must be non-empty")
            decoded = json.loads(raw_text)
            normalized = _validate_structured_payload(decoded)
            return raw_text, normalized
    raise ValueError("missing output_text")


def _validate_structured_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("structured payload must be an object")
    if set(payload) != {"reply", "supported_evidence_ids", "handoff", "reason"}:
        raise ValueError("structured payload has wrong keys")
    if not isinstance(payload["reply"], str):
        raise ValueError("reply must be a string")
    supported = payload["supported_evidence_ids"]
    if not isinstance(supported, list) or any(
        not isinstance(item, str) for item in supported
    ):
        raise ValueError("supported_evidence_ids must be a list of strings")
    if not isinstance(payload["handoff"], bool):
        raise ValueError("handoff must be a bool")
    if not isinstance(payload["reason"], str):
        raise ValueError("reason must be a string")
    return {
        "reply": payload["reply"],
        "supported_evidence_ids": list(supported),
        "handoff": payload["handoff"],
        "reason": payload["reason"],
    }


def _parse_usage(payload: object) -> tuple[int, int]:
    if not isinstance(payload, Mapping):
        return 0, 0
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return 0, 0
    return _read_usage_int(usage, "input_tokens"), _read_usage_int(
        usage, "output_tokens"
    )


def _read_usage_int(usage: Mapping[str, object], name: str) -> int:
    value = usage.get(name)
    if isinstance(value, bool):
        return 0
    if not isinstance(value, int):
        return 0
    return value


def validate_api_base_url(value: str) -> str:
    """Validate without echoing potentially secret URL contents."""
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
        if not valid:
            raise ValueError
        parsed.port
    except (ValueError, TypeError, AttributeError):
        raise ValueError("API base URL must be HTTPS without credentials, query or fragment") from None
    return value.rstrip("/")


def _authorization_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _require_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
