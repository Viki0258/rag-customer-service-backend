from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from whatsapp_support.domain import ModelRequest, RetrievedChunk, StructuredModelResult
from whatsapp_support.memory import ConversationContext
from whatsapp_support.meta_whatsapp import AsyncHttpTransport, HttpResponse
from whatsapp_support.openai_adapters import (
    OpenAIAdapterError,
    OpenAIEmbeddingAdapter,
    OpenAIResponseAdapter,
)
from whatsapp_support.rag import RagPromptBuilder


API_KEY = "super-secret-openai-key"


@pytest.mark.asyncio
async def test_separate_provider_files_route_keys_and_models(tmp_path, monkeypatch):
    from whatsapp_support.config import RagSettings

    embedding = tmp_path / "embedding.json"
    response = tmp_path / "response.json"
    embedding.write_text(json.dumps({"api_key": "embedding-secret", "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1/", "embedding_model": "text-embedding-v4", "dimensions": 2}))
    response.write_text(json.dumps({"api_key": "reply-secret", "base_url": "https://api.deepseek.com/", "response_model": "deepseek-flash"}))
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_MODE", "compatible")
    monkeypatch.setenv("WHATSAPP_SUPPORT_EMBEDDING_CONFIG_FILE", str(embedding))
    monkeypatch.setenv("WHATSAPP_SUPPORT_RESPONSE_CONFIG_FILE", str(response))
    settings = RagSettings.from_env()
    assert "embedding-secret" not in repr(settings)
    assert "reply-secret" not in repr(settings)
    assert settings.index_path.endswith("bailian.sqlite3")
    et = ScriptedTransport(HttpResponse(200, {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(10)]}), HttpResponse(200, {"data": [{"index": 0, "embedding": [10.0, 1.0]}]}))
    vectors = await settings.create_embedding(transport_factory=lambda: et).embed([str(i) for i in range(11)])
    assert vectors == [[float(i), 1.0] for i in range(11)]
    assert [len(c.json_body["input"]) for c in et.calls] == [10, 1]
    assert all(c.url == "https://maas.qianwenaiapi.com/compatible-mode/v1/embeddings" for c in et.calls)
    assert all(c.headers["Authorization"] == "Bearer embedding-secret" for c in et.calls)
    payload = {"reply": "Shipping takes 3 days.", "supported_evidence_ids": ["chunk-1"], "handoff": False, "reason": "supported"}
    rt = ScriptedTransport(HttpResponse(200, {"output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}], "usage": {"input_tokens": 10, "output_tokens": 20}}))
    result = await settings.create_model(transport_factory=lambda: rt).generate(make_request())
    assert result.content == payload
    assert result.tokens_in == 10
    assert rt.calls[0].url == "https://api.deepseek.com/responses"
    assert rt.calls[0].headers["Authorization"] == "Bearer reply-secret"
    assert rt.calls[0].json_body["model"] == "deepseek-flash"
    assert rt.calls[0].json_body.get("stream", False) is False
    assert rt.calls[0].json_body["text"]["format"]["strict"] is True


@pytest.mark.parametrize("url", ["http://api.deepseek.com", "https://user:secret@api.deepseek.com", "https://api.deepseek.com?key=secret", "https://api.deepseek.com#secret", "not-a-url"])
def test_provider_base_url_rejects_unsafe_config_without_echoing(url):
    from whatsapp_support.openai_adapters import validate_api_base_url
    with pytest.raises(ValueError) as error:
        validate_api_base_url(url)
    assert "secret" not in str(error.value)


def test_provider_configuration_errors_do_not_echo_secrets(tmp_path):
    from whatsapp_support.config import RagSettings
    path = tmp_path / "bad.json"
    path.write_text('{"api_key":"private-secret", INVALID}')
    with pytest.raises(ValueError) as error:
        RagSettings.from_provider_files(str(path), str(path))
    assert "private-secret" not in str(error.value)

EMBEDDING_MODEL = "text-embedding-3-large"
RESPONSE_MODEL = "gpt-5.6-mini"
REQUESTED_AT = datetime(2026, 9, 5, 8, 30, tzinfo=UTC)
STRUCTURED_SCHEMA = {
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


@dataclass(frozen=True)
class RecordedCall:
    url: str
    headers: dict[str, str]
    json_body: dict[str, object]
    timeout: float


class ScriptedTransport:
    def __init__(self, *outcomes: HttpResponse | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[RecordedCall] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
        timeout: float,
    ) -> HttpResponse:
        self.calls.append(
            RecordedCall(url, dict(headers), dict(json_body), timeout)
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True


def make_request() -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        prompt_name="rag-answer",
        prompt_version="v3",
        inputs={
            "messages": [
                {"role": "system", "content": "Use only supplied evidence."},
                {"role": "user", "content": "When does shipping arrive?"},
            ],
            "locale": "en",
            "audit": {"trace_id": "trace-1", "conversation_id": "conversation-1"},
            "evidence": [
                {"chunk_id": "chunk-1", "text": "Shipping takes 3 days."}
            ],
        },
        trace_id="trace-1",
        requested_at=REQUESTED_AT,
    )


@pytest.mark.asyncio
async def test_embedding_adapter_posts_batch_request_and_returns_ordered_vectors() -> None:
    transport = ScriptedTransport(
        HttpResponse(
            200,
                {
                    "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                    ]
                },
            )
    )
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        transport,
        timeout_seconds=4.5,
        max_retries=2,
    )

    result = await adapter.embed(["first", "second"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert adapter.model_name == EMBEDDING_MODEL
    assert adapter.dimension == 3
    assert transport.calls == [
        RecordedCall(
            "https://api.openai.com/v1/embeddings",
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            {
                "model": EMBEDDING_MODEL,
                "input": ["first", "second"],
                "dimensions": 3,
            },
            4.5,
        )
    ]


@pytest.mark.asyncio
async def test_embedding_adapter_restores_input_order_from_response_indexes() -> None:
    transport = ScriptedTransport(
        HttpResponse(
            200,
            {
                "data": [
                    {"index": 2, "embedding": [2.1, 2.2, 2.3]},
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [1.1, 1.2, 1.3]},
                ]
            },
        )
    )
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        transport,
        timeout_seconds=4.5,
        max_retries=2,
    )

    result = await adapter.embed(["first", "second", "third"])

    assert result == [
        [0.1, 0.2, 0.3],
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
    ]
    assert len(transport.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "texts"),
    [
        pytest.param(
            [{"embedding": [0.1, 0.2, 0.3]}],
            ["first"],
            id="missing-index",
        ),
        pytest.param(
            [{"index": "0", "embedding": [0.1, 0.2, 0.3]}],
            ["first"],
            id="string-index",
        ),
        pytest.param(
            [{"index": True, "embedding": [0.1, 0.2, 0.3]}],
            ["first"],
            id="bool-index",
        ),
        pytest.param(
            [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 0, "embedding": [1.1, 1.2, 1.3]},
            ],
            ["first", "second"],
            id="duplicate-index",
        ),
        pytest.param(
            [{"index": -1, "embedding": [0.1, 0.2, 0.3]}],
            ["first"],
            id="negative-index",
        ),
        pytest.param(
            [{"index": 1, "embedding": [0.1, 0.2, 0.3]}],
            ["first"],
            id="out-of-range-index",
        ),
        pytest.param(
            [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 2, "embedding": [2.1, 2.2, 2.3]},
            ],
            ["first", "second", "third"],
            id="non-contiguous-indexes",
        ),
        pytest.param(
            [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            ["first", "second"],
            id="row-count-too-small",
        ),
        pytest.param(
            [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [1.1, 1.2, 1.3]},
                {"index": 2, "embedding": [2.1, 2.2, 2.3]},
            ],
            ["first", "second"],
            id="row-count-too-large",
        ),
    ],
)
async def test_embedding_adapter_fails_without_retry_on_invalid_embedding_indexes(
    rows: list[dict[str, object]],
    texts: list[str],
) -> None:
    transport = ScriptedTransport(HttpResponse(200, {"data": rows}))
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        transport,
        timeout_seconds=6.0,
        max_retries=3,
    )

    with pytest.raises(OpenAIAdapterError) as excinfo:
        await adapter.embed(texts)

    assert excinfo.value.error_code == "embedding_failed"
    assert excinfo.value.retryable is False
    assert str(excinfo.value) == "embedding_failed"
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_embedding_adapter_retries_retryable_transport_and_http_failures() -> None:
    transport = ScriptedTransport(
        ConnectionError(f"network leaked {API_KEY}"),
        HttpResponse(429, {"error": {"message": f"retry leaked {API_KEY}"}}),
        HttpResponse(200, {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}),
    )
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        transport,
        timeout_seconds=6.0,
        max_retries=2,
    )

    result = await adapter.embed(["retry me"])

    assert result == [[1.0, 2.0, 3.0]]
    assert len(transport.calls) == 3


@pytest.mark.asyncio
async def test_embedding_adapter_does_not_retry_bad_request_or_dimension_mismatch() -> None:
    bad_request = ScriptedTransport(
        HttpResponse(400, {"error": {"message": f"bad request {API_KEY}"}})
    )
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        bad_request,
        timeout_seconds=6.0,
        max_retries=3,
    )

    with pytest.raises(OpenAIAdapterError) as bad_request_error:
        await adapter.embed(["bad request"])

    assert bad_request_error.value.error_code == "embedding_failed"
    assert bad_request_error.value.retryable is False
    assert str(bad_request_error.value) == "embedding_failed"
    assert API_KEY not in repr(bad_request_error.value)
    assert len(bad_request.calls) == 1

    wrong_dimension = ScriptedTransport(
        HttpResponse(200, {"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
    )
    dimension_adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        wrong_dimension,
        timeout_seconds=6.0,
        max_retries=3,
    )

    with pytest.raises(OpenAIAdapterError) as dimension_error:
        await dimension_adapter.embed(["wrong dimension"])

    assert dimension_error.value.error_code == "embedding_failed"
    assert dimension_error.value.retryable is False
    assert len(wrong_dimension.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(True, id="bool"),
        pytest.param("0.1", id="numeric-string"),
        pytest.param(1 + 0j, id="complex"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
async def test_embedding_adapter_rejects_non_finite_or_non_real_vector_values(
    invalid_value: object,
) -> None:
    transport = ScriptedTransport(
        HttpResponse(
            200,
            {"data": [{"index": 0, "embedding": [invalid_value, 0.2, 0.3]}]},
        )
    )
    adapter = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        transport,
        timeout_seconds=6.0,
        max_retries=3,
    )

    with pytest.raises(OpenAIAdapterError) as excinfo:
        await adapter.embed(["malformed vector"])

    assert excinfo.value.error_code == "embedding_failed"
    assert excinfo.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_response_adapter_posts_json_schema_and_parses_structured_result() -> None:
    raw_text = json.dumps(
        {
            "reply": "Shipping usually arrives in 3 business days.",
            "supported_evidence_ids": ["chunk-1"],
            "handoff": False,
            "reason": "grounded_answer",
        }
    )
    transport = ScriptedTransport(
        HttpResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": raw_text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 45,
                },
            },
        )
    )
    adapter = OpenAIResponseAdapter(
        API_KEY,
        RESPONSE_MODEL,
        transport,
        timeout_seconds=9.0,
        max_retries=1,
    )
    request = make_request()

    result = await adapter.generate(request)

    assert isinstance(result, StructuredModelResult)
    assert result.trace_id == request.trace_id
    assert result.model_name == RESPONSE_MODEL
    assert result.prompt_version == request.prompt_version
    assert result.content == {
        "reply": "Shipping usually arrives in 3 business days.",
        "supported_evidence_ids": ["chunk-1"],
        "handoff": False,
        "reason": "grounded_answer",
    }
    assert result.raw_text == raw_text
    assert result.tokens_in == 120
    assert result.tokens_out == 45
    assert result.created_at.tzinfo == UTC
    assert transport.calls == [
        RecordedCall(
            "https://api.openai.com/v1/responses",
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            {
                "model": RESPONSE_MODEL,
                "input": [
                    {"role": "system", "content": "Use only supplied evidence."},
                    {"role": "user", "content": "When does shipping arrive?"},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "rag_answer",
                        "schema": STRUCTURED_SCHEMA,
                        "strict": True,
                    }
                },
            },
            9.0,
        )
    ]


@pytest.mark.asyncio
async def test_response_adapter_retries_retryable_failures_and_stops_on_schema_errors() -> None:
    request = make_request()
    valid_raw_text = json.dumps(
        {
            "reply": "3 business days.",
            "supported_evidence_ids": ["chunk-1"],
            "handoff": False,
            "reason": "grounded_answer",
        }
    )
    retrying_transport = ScriptedTransport(
        TimeoutError(f"timeout leaked {API_KEY}"),
        HttpResponse(
            503,
            {"error": {"message": f"server unavailable leaked {API_KEY}"}},
        ),
        HttpResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": valid_raw_text}],
                    }
                ]
            },
        ),
    )
    adapter = OpenAIResponseAdapter(
        API_KEY,
        RESPONSE_MODEL,
        retrying_transport,
        timeout_seconds=8.0,
        max_retries=2,
    )

    result = await adapter.generate(request)

    assert result.content["reply"] == "3 business days."
    assert len(retrying_transport.calls) == 3

    invalid_schema_transport = ScriptedTransport(
        HttpResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "reply": "Oops",
                                        "handoff": False,
                                        "reason": "grounded_answer",
                                    }
                                ),
                            }
                        ],
                    }
                ]
            },
        )
    )
    invalid_adapter = OpenAIResponseAdapter(
        API_KEY,
        RESPONSE_MODEL,
        invalid_schema_transport,
        timeout_seconds=8.0,
        max_retries=2,
    )

    with pytest.raises(OpenAIAdapterError) as excinfo:
        await invalid_adapter.generate(request)

    assert excinfo.value.error_code == "model_schema_invalid"
    assert excinfo.value.retryable is False
    assert str(excinfo.value) == "model_schema_invalid"
    assert API_KEY not in repr(excinfo.value)
    assert len(invalid_schema_transport.calls) == 1


@pytest.mark.asyncio
async def test_openai_adapters_aclose_closes_injected_transport() -> None:
    embedding_transport = ScriptedTransport(
        HttpResponse(200, {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})
    )
    response_transport = ScriptedTransport(
        HttpResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "reply": "ok",
                                        "supported_evidence_ids": ["chunk-1"],
                                        "handoff": False,
                                        "reason": "grounded_answer",
                                    }
                                ),
                            }
                        ],
                    }
                ]
            },
        )
    )
    embedding = OpenAIEmbeddingAdapter(
        API_KEY,
        EMBEDDING_MODEL,
        3,
        embedding_transport,
        timeout_seconds=5.0,
        max_retries=0,
    )
    response = OpenAIResponseAdapter(
        API_KEY,
        RESPONSE_MODEL,
        response_transport,
        timeout_seconds=5.0,
        max_retries=0,
    )

    await embedding.aclose()
    await response.aclose()

    assert embedding_transport.closed is True
    assert response_transport.closed is True


@pytest.mark.asyncio
async def test_response_adapter_maps_real_rag_prompt_without_sending_audit() -> None:
    request = RagPromptBuilder().build(
        context=ConversationContext(None, (), "none"),
        question="When does shipping arrive?",
        retrieved_chunks=[RetrievedChunk(
            chunk_id="chunk-1", document_id="doc-1", locale="en",
            text="Shipping takes 3 days.", vector_rank=1, keyword_rank=1,
            fused_score=0.8, metadata={},
        )],
        locale="en", prompt_version="v3", conversation_id="conversation-1",
        trace_id="trace-1", requested_at=REQUESTED_AT,
    )
    transport = ScriptedTransport(HttpResponse(200, {
        "output": [{"type": "message", "role": "assistant", "content": [{
            "type": "output_text", "text": json.dumps({
                "reply": "Shipping takes 3 days.", "supported_evidence_ids": ["chunk-1"],
                "handoff": False, "reason": "grounded_answer",
            }),
        }]}],
    }))
    adapter = OpenAIResponseAdapter(
        API_KEY, RESPONSE_MODEL, transport, timeout_seconds=9.0, max_retries=0,
    )

    result = await adapter.generate(request)

    body = transport.calls[0].json_body
    messages = body["input"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "When does shipping arrive?" in messages[1]["content"]
    references = json.loads(messages[1]["content"].split("UNTRUSTED REFERENCE DATA:\n", 1)[1])
    assert references[0]["chunk_id"] == "chunk-1"
    assert references[0]["text"] == "Shipping takes 3 days."
    assert "trace-1" not in json.dumps(body)
    assert "conversation-1" not in json.dumps(body)
    assert "audit" not in body
    assert request.inputs["audit"]["trace_id"] == "trace-1"
    assert result.trace_id == "trace-1"
    assert result.prompt_version == "v3"
