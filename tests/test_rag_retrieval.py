from __future__ import annotations

import pytest

from whatsapp_support.domain import RetrievedChunk
import whatsapp_support.rag_retrieval as rag_retrieval
from whatsapp_support.rag_retrieval import (
    HybridRetriever,
    compatible_locales,
    cosine_similarity,
    normalize_locale,
    rrf_merge,
    tokenize_for_bm25,
)


def make_chunk(
    chunk_id: str,
    *,
    locale: str,
    text: str,
    vector_rank: int,
    keyword_rank: int,
    fused_score: float,
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        locale=locale,
        text=text,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        fused_score=fused_score,
        metadata={} if metadata is None else metadata,
    )


def test_normalize_locale_canonicalizes_case_and_separators() -> None:
    assert normalize_locale(" zh_cn ") == "zh-CN"
    assert normalize_locale("EN-us") == "en-US"


def test_compatible_locales_keeps_same_language_only() -> None:
    assert compatible_locales("zh-CN") == frozenset({"zh-CN", "zh"})
    assert compatible_locales("en-US") == frozenset({"en-US", "en"})


def test_tokenize_for_bm25_uses_chinese_ngrams_and_preserves_identifiers() -> None:
    tokens = tokenize_for_bm25("退货政策支持7天退款，SKU A12-34，v1.2.3", "zh-CN")

    assert tokens.count("退货") == 1
    assert tokens.count("政策") == 1
    assert tokens.count("退款") == 1
    assert "sku" in tokens
    assert "a12-34" in tokens
    assert "7" in tokens
    assert "v1.2.3" in tokens


def test_tokenize_for_bm25_preserves_underscored_identifiers() -> None:
    tokens = tokenize_for_bm25("foo_bar SKU_123 v2_0", "en")

    assert "foo_bar" in tokens
    assert "sku_123" in tokens
    assert "v2_0" in tokens


def test_tokenize_for_bm25_uses_unicode_words_in_english() -> None:
    tokens = tokenize_for_bm25("Café naïve façade SKU A12-34 and v1.2.3", "en")

    assert tokens == ["café", "naïve", "façade", "sku", "a12-34", "and", "v1.2.3"]


def test_cosine_similarity_handles_zero_vectors_and_orders_scores() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) > cosine_similarity(
        [1.0, 0.0], [0.0, 1.0]
    )


def test_rrf_merge_deduplicates_by_chunk_id_and_applies_limit() -> None:
    vector_hits = [
        make_chunk("chunk-1", locale="en", text="a", vector_rank=1, keyword_rank=0, fused_score=0.0),
        make_chunk("chunk-2", locale="en", text="b", vector_rank=2, keyword_rank=0, fused_score=0.0),
    ]
    keyword_hits = [
        make_chunk("chunk-1", locale="en", text="a", vector_rank=9, keyword_rank=1, fused_score=0.0),
        make_chunk("chunk-3", locale="en", text="c", vector_rank=0, keyword_rank=2, fused_score=0.0),
    ]

    merged = rrf_merge(vector_hits, keyword_hits, limit=2)

    assert [chunk.chunk_id for chunk in merged] == ["chunk-1", "chunk-2"]
    assert merged[0].fused_score > merged[1].fused_score


def test_hybrid_retriever_filters_locale_before_ranking_and_applies_threshold() -> None:
    chunks = [
        make_chunk(
            "zh-1",
            locale="zh-CN",
            text="退货政策支持7天退货",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [1.0, 0.0]},
        ),
        make_chunk(
            "en-1",
            locale="en",
            text="Return policy supports 7-day returns",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [0.0, 1.0]},
        ),
    ]

    retriever = HybridRetriever()
    results = retriever.search(
        chunks=chunks,
        query_vector=[1.0, 0.0],
        query_text="退货政策",
        locale="zh",
        candidate_k=2,
        final_k=5,
        threshold=0.0,
    )

    assert [chunk.chunk_id for chunk in results] == ["zh-1"]


def test_hybrid_retriever_returns_only_hits_meeting_threshold() -> None:
    chunks = [
        make_chunk(
            "en-1",
            locale="en",
            text="apple",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [1.0, 0.0]},
        ),
        make_chunk(
            "en-2",
            locale="en",
            text="banana",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [0.0, 1.0]},
        ),
    ]

    retriever = HybridRetriever()
    results = retriever.search(
        chunks=chunks,
        query_vector=[1.0, 0.0],
        query_text="apple",
        locale="en",
        candidate_k=2,
        final_k=5,
        threshold=0.03,
    )

    assert [chunk.chunk_id for chunk in results] == ["en-1"]


def test_hybrid_retriever_honors_independent_candidate_limits() -> None:
    chunks = [
        make_chunk(
            "vector-only",
            locale="en",
            text="backpack material",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [1.0, 0.0]},
        ),
        make_chunk(
            "keyword-1",
            locale="en",
            text="return return policy",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [0.8, 0.2]},
        ),
        make_chunk(
            "keyword-2",
            locale="en",
            text="return policy details",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [0.0, 1.0]},
        ),
    ]

    results = HybridRetriever().search(
        chunks=chunks,
        query_vector=[1.0, 0.0],
        query_text="return policy",
        locale="en",
        candidate_k=3,
        vector_candidate_k=1,
        bm25_candidate_k=2,
        final_k=3,
        threshold=0.0,
    )

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["vector-only"].vector_rank == 1
    assert by_id["vector-only"].keyword_rank == 0
    assert by_id["keyword-1"].vector_rank == 0
    assert by_id["keyword-1"].keyword_rank == 1
    assert by_id["keyword-2"].vector_rank == 0
    assert by_id["keyword-2"].keyword_rank == 2


def test_hybrid_retriever_keeps_lower_ranked_qualifier_after_threshold() -> None:
    chunks = [
        make_chunk(
            "en-1",
            locale="en",
            text="apple",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [1.0, 0.0]},
        ),
        make_chunk(
            "en-2",
            locale="en",
            text="banana",
            vector_rank=0,
            keyword_rank=0,
            fused_score=0.0,
            metadata={"embedding": [0.0, 1.0]},
        ),
    ]

    vector_hits = [
        make_chunk("bad-1", locale="en", text="bad", vector_rank=1, keyword_rank=0, fused_score=0.0),
    ]
    keyword_hits = [
        make_chunk("good-1", locale="en", text="good", vector_rank=0, keyword_rank=1, fused_score=0.0),
    ]
    merged_ranking = [
        make_chunk("bad-1", locale="en", text="bad", vector_rank=1, keyword_rank=1, fused_score=0.01),
        make_chunk("good-1", locale="en", text="good", vector_rank=2, keyword_rank=2, fused_score=0.03),
    ]

    retriever = HybridRetriever()

    def fake_rrf_merge(vector_hits, keyword_hits, limit):
        return merged_ranking[:limit]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rag_retrieval, "rrf_merge", fake_rrf_merge)
    monkeypatch.setattr(retriever, "_vector_hits", lambda *args, **kwargs: vector_hits)
    monkeypatch.setattr(retriever, "_keyword_hits", lambda *args, **kwargs: keyword_hits)
    try:
        results = retriever.search(
            chunks=chunks,
            query_vector=[1.0, 0.0],
            query_text="good",
            locale="en",
            candidate_k=2,
            final_k=1,
            threshold=0.02,
        )
    finally:
        monkeypatch.undo()

    assert [chunk.chunk_id for chunk in results] == ["good-1"]
