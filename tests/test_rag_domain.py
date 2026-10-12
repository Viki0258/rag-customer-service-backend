from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import get_args, get_origin

import pytest

from whatsapp_support.domain import (
    ConversationSummary,
    ConversationTurn,
    Evidence,
    RagAnswerResult,
    RetrievedChunk,
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


def test_retrieved_chunk_requires_utc_timestamp_and_is_frozen() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        locale="en",
        text="hello",
        vector_rank=1,
        keyword_rank=2,
        fused_score=0.75,
        metadata={"source": "kb"},
    )

    assert is_dataclass(chunk)
    with pytest.raises(FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]


def test_retrieved_chunk_rejects_empty_chunk_id() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        RetrievedChunk(
            chunk_id=" ",
            document_id="doc-1",
            locale="en",
            text="hello",
            vector_rank=1,
            keyword_rank=2,
            fused_score=0.75,
            metadata={},
        )


def test_conversation_turn_requires_non_blank_text_and_keys() -> None:
    with pytest.raises(ValueError, match="customer_text"):
        ConversationTurn(
            customer_text=" ",
            assistant_text="reply",
            event_key="evt-1",
            dedup_key="dedup-1",
        )


def test_rag_answer_result_accepts_only_known_status_values() -> None:
    result = RagAnswerResult(
        status="reply",
        reply="ok",
        locale="en",
        trace_id="trace-1",
        retrieved_chunk_ids=["chunk-1"],
        supported_chunk_ids=("chunk-1",),
        prompt_version="v1",
        model_name="model-1",
        reason="grounded_answer",
    )

    assert result.status == "reply"
    with pytest.raises(ValueError, match="status"):
        RagAnswerResult(
            status="maybe",
            reply="ok",
            locale="en",
            trace_id="trace-1",
            retrieved_chunk_ids=(),
            supported_chunk_ids=(),
            prompt_version="v1",
            model_name="model-1",
            reason=None,
        )


def test_rag_answer_result_normalizes_evidence_ids_to_tuples() -> None:
    result = RagAnswerResult(
        status="handoff",
        reply=None,
        locale="zh-CN",
        trace_id="trace-2",
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        supported_chunk_ids=["chunk-1"],
        prompt_version="v1",
        model_name="model-1",
        reason="insufficient_evidence",
    )

    assert result.retrieved_chunk_ids == ("chunk-1", "chunk-2")
    assert result.supported_chunk_ids == ("chunk-1",)


def test_conversation_summary_requires_dict_data_shape() -> None:
    summary = ConversationSummary(
        version=1,
        covered_until=utc_datetime(2026, 9, 1, 9, 0, 0),
        data={
            "customer_goal": "return item",
            "confirmed_entities": [],
            "customer_preferences": [],
            "unresolved_questions": [],
            "decisions": [],
            "handoff_state": "none",
        },
    )

    assert summary.data["handoff_state"] == "none"
    with pytest.raises(ValueError, match="data"):
        ConversationSummary(
            version=1,
            covered_until=utc_datetime(2026, 9, 1, 9, 0, 0),
            data=[],
        )  # type: ignore[arg-type]


def test_retrieved_chunk_is_distinct_from_evidence() -> None:
    assert RetrievedChunk is not Evidence
    assert RetrievedChunk.__name__ == "RetrievedChunk"
    assert Evidence.__name__ == "Evidence"
    assert isinstance(
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            locale="en",
            text="text",
            vector_rank=1,
            keyword_rank=2,
            fused_score=1.0,
            metadata={},
        ),
        RetrievedChunk,
    )
    assert not isinstance(
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            locale="en",
            text="text",
            vector_rank=1,
            keyword_rank=2,
            fused_score=1.0,
            metadata={},
        ),
        Evidence,
    )


def test_ports_protocols_expose_expected_async_methods() -> None:
    from whatsapp_support.ports import (
        ConversationMemoryPort,
        HybridKnowledgePort,
        RagAnswerPort,
        SummaryModelPort,
    )

    memory_methods = {
        name
        for name, member in vars(ConversationMemoryPort).items()
        if callable(member) and not name.startswith("_")
    }

    assert memory_methods == {"get_recent_turns", "get_summary", "save_summary"}
    assert hasattr(HybridKnowledgePort, "retrieve")
    assert hasattr(RagAnswerPort, "answer")
    assert hasattr(SummaryModelPort, "summarize")
