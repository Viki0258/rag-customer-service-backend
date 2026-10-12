from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Sequence

from whatsapp_support.domain import RetrievedChunk


_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[^\W]+(?:[._-][^\W]+)*", re.UNICODE)
_SPAN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|[^\u3400-\u4dbf\u4e00-\u9fff]+"
)


def normalize_locale(locale: str) -> str:
    if not isinstance(locale, str) or not locale.strip():
        raise ValueError("locale must be a non-empty string")
    parts = locale.strip().replace("_", "-").split("-")
    language = parts[0].lower()
    if len(parts) == 1:
        return language
    region = parts[1].upper()
    extras = [part.upper() for part in parts[2:]]
    return "-".join([language, region, *extras])


def compatible_locales(requested: str) -> frozenset[str]:
    normalized = normalize_locale(requested)
    base = normalized.split("-", 1)[0]
    return frozenset({normalized, base})


def tokenize_for_bm25(text: str, locale: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized_locale = normalize_locale(locale)
    return (
        _tokenize_chinese(text)
        if normalized_locale.startswith("zh")
        else _tokenize_word_text(text)
    )


def cosine_similarity(query: Sequence[float], candidate: Sequence[float]) -> float:
    if len(query) != len(candidate):
        raise ValueError("query and candidate must have the same dimensions")
    numerator = sum(left * right for left, right in zip(query, candidate, strict=True))
    query_norm = math.sqrt(sum(value * value for value in query))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    if query_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return numerator / (query_norm * candidate_norm)


def rrf_merge(
    vector_hits: Sequence[RetrievedChunk],
    keyword_hits: Sequence[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    if limit < 1:
        return []

    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for hit in vector_hits:
        score = 1.0 / (60 + hit.vector_rank)
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + score
        merged[hit.chunk_id] = hit

    for hit in keyword_hits:
        score = 1.0 / (60 + hit.keyword_rank)
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + score
        merged[hit.chunk_id] = _prefer_more_complete_chunk(merged.get(hit.chunk_id), hit)

    ranked = sorted(
        (
            replace(chunk, fused_score=scores[chunk_id])
            for chunk_id, chunk in merged.items()
        ),
        key=lambda chunk: (
            -chunk.fused_score,
            _sort_rank(chunk.vector_rank),
            _sort_rank(chunk.keyword_rank),
            chunk.chunk_id,
        ),
    )
    return ranked[:limit]


class HybridRetriever:
    def search(
        self,
        chunks: Sequence[RetrievedChunk],
        query_vector: Sequence[float],
        query_text: str,
        locale: str,
        candidate_k: int,
        final_k: int,
        threshold: float,
        vector_candidate_k: int | None = None,
        bm25_candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if candidate_k < 1 or final_k < 1:
            return []

        requested = normalize_locale(locale)
        eligible_chunks = [
            chunk for chunk in chunks if _locale_matches(requested, normalize_locale(chunk.locale))
        ]
        if not eligible_chunks:
            return []

        vector_hits = self._vector_hits(
            eligible_chunks, query_vector, vector_candidate_k or candidate_k
        )
        keyword_hits = self._keyword_hits(
            eligible_chunks, query_text, requested, bm25_candidate_k or candidate_k
        )
        merged = rrf_merge(
            vector_hits,
            keyword_hits,
            limit=len(vector_hits) + len(keyword_hits),
        )
        filtered = [chunk for chunk in merged if chunk.fused_score >= threshold]
        return filtered[:final_k]

    def _vector_hits(
        self,
        chunks: Sequence[RetrievedChunk],
        query_vector: Sequence[float],
        candidate_k: int,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            score = cosine_similarity(query_vector, _embedding_from_chunk(chunk))
            if score > 0.0:
                scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:candidate_k]
        return [
            replace(chunk, vector_rank=index, keyword_rank=0)
            for index, (_, chunk) in enumerate(ranked, start=1)
        ]

    def _keyword_hits(
        self,
        chunks: Sequence[RetrievedChunk],
        query_text: str,
        locale: str,
        candidate_k: int,
    ) -> list[RetrievedChunk]:
        query_tokens = tokenize_for_bm25(query_text, locale)
        if not query_tokens:
            return []

        corpus_tokens = [tokenize_for_bm25(chunk.text, chunk.locale) for chunk in chunks]
        document_count = len(corpus_tokens)
        avg_doc_len = sum(len(tokens) for tokens in corpus_tokens) / document_count
        doc_frequency: Counter[str] = Counter()
        for tokens in corpus_tokens:
            doc_frequency.update(set(tokens))

        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk, tokens in zip(chunks, corpus_tokens, strict=True):
            score = _bm25_score(query_tokens, tokens, doc_frequency, document_count, avg_doc_len)
            if score > 0.0:
                scored.append((score, chunk))

        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:candidate_k]
        return [
            replace(chunk, keyword_rank=index, vector_rank=0)
            for index, (_, chunk) in enumerate(ranked, start=1)
        ]


def _tokenize_chinese(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _SPAN_RE.finditer(text):
        span = match.group(0)
        if _HAN_RE.fullmatch(span):
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
            continue
        tokens.extend(_tokenize_word_text(span))
    return tokens


def _tokenize_word_text(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _locale_matches(requested: str, candidate: str) -> bool:
    requested_base = requested.split("-", 1)[0]
    candidate_base = candidate.split("-", 1)[0]
    if requested == candidate:
        return True
    if requested == requested_base:
        return candidate_base == requested_base
    return candidate in compatible_locales(requested)


def _embedding_from_chunk(chunk: RetrievedChunk) -> Sequence[float]:
    embedding = chunk.metadata.get("embedding")
    if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
        return ()
    vector: list[float] = []
    for value in embedding:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return ()
        vector.append(float(value))
    return vector


def _bm25_score(
    query_tokens: Sequence[str],
    document_tokens: Sequence[str],
    doc_frequency: Counter[str],
    document_count: int,
    avg_doc_len: float,
) -> float:
    if not document_tokens or avg_doc_len == 0.0:
        return 0.0

    frequencies = Counter(document_tokens)
    score = 0.0
    k1 = 1.5
    b = 0.75
    doc_len = len(document_tokens)

    for token in query_tokens:
        term_frequency = frequencies[token]
        if term_frequency == 0:
            continue
        df = doc_frequency[token]
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        numerator = term_frequency * (k1 + 1.0)
        denominator = term_frequency + k1 * (1.0 - b + b * doc_len / avg_doc_len)
        score += idf * (numerator / denominator)
    return score


def _prefer_more_complete_chunk(
    existing: RetrievedChunk | None, candidate: RetrievedChunk
) -> RetrievedChunk:
    if existing is None:
        return candidate
    if existing.vector_rank == 0 and candidate.vector_rank != 0:
        return replace(candidate, keyword_rank=existing.keyword_rank or candidate.keyword_rank)
    if existing.keyword_rank == 0 and candidate.keyword_rank != 0:
        return replace(existing, keyword_rank=candidate.keyword_rank)
    return existing


def _sort_rank(rank: int) -> int:
    return rank if rank > 0 else 10**9
