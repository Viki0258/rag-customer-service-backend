from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from whatsapp_support.meta_whatsapp import AsyncHttpTransport


DATABASE_URL_ENV = "WHATSAPP_SUPPORT_DATABASE_URL"
REDIS_URL_ENV = "WHATSAPP_SUPPORT_REDIS_URL"
STREAM_NAME_ENV = "WHATSAPP_SUPPORT_REDIS_STREAM"
CONSUMER_GROUP_ENV = "WHATSAPP_SUPPORT_REDIS_GROUP"
LEASE_SECONDS_ENV = "WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS"
MAX_ATTEMPTS_ENV = "WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS"
PROVIDER_ENV = "WHATSAPP_SUPPORT_PROVIDER"
META_ACCESS_TOKEN_ENV = "WHATSAPP_SUPPORT_META_ACCESS_TOKEN"
META_PHONE_NUMBER_ID_ENV = "WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID"
META_GRAPH_BASE_URL_ENV = "WHATSAPP_SUPPORT_META_GRAPH_BASE_URL"
RAG_MODE_ENV = "WHATSAPP_SUPPORT_RAG_MODE"
RAG_INDEX_PATH_ENV = "WHATSAPP_SUPPORT_RAG_INDEX_PATH"
RAG_EMBEDDING_DIMENSION_ENV = "WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION"
RAG_EMBEDDING_TIMEOUT_ENV = "WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS"
RAG_EMBEDDING_RETRIES_ENV = "WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES"
RAG_RESPONSE_TIMEOUT_ENV = "WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS"
RAG_RESPONSE_RETRIES_ENV = "WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES"
RAG_VECTOR_CANDIDATES_ENV = "WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES"
RAG_BM25_CANDIDATES_ENV = "WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES"
RAG_FINAL_CANDIDATES_ENV = "WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES"
RAG_EVIDENCE_THRESHOLD_ENV = "WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD"
RAG_SUMMARY_TRIGGER_ENV = "WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS"
RAG_SUMMARY_RETENTION_ENV = "WHATSAPP_SUPPORT_RAG_SUMMARY_RETENTION_TURNS"
RAG_REPLY_MAX_CHARS_ENV = "WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS"
RAG_PROMPT_VERSION_ENV = "WHATSAPP_SUPPORT_RAG_PROMPT_VERSION"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
OPENAI_RESPONSE_MODEL_ENV = "OPENAI_RESPONSE_MODEL"

_LEGACY_STREAM_NAME_ENV = "WHATSAPP_SUPPORT_STREAM_NAME"
_LEGACY_CONSUMER_GROUP_ENV = "WHATSAPP_SUPPORT_CONSUMER_GROUP"
_LEGACY_LEASE_SECONDS_ENV = "WHATSAPP_SUPPORT_LEASE_SECONDS"
_LEGACY_MAX_ATTEMPTS_ENV = "WHATSAPP_SUPPORT_MAX_ATTEMPTS"


@dataclass(frozen=True)
class WhatsAppProviderSettings:
    mode: Literal["fake", "meta_cloud"] = "fake"
    meta_access_token: str | None = None
    meta_phone_number_id: str | None = None
    meta_graph_base_url: str | None = None

    @property
    def is_complete(self) -> bool:
        return all(
            isinstance(value, str) and bool(value.strip())
            for value in (
                self.meta_access_token,
                self.meta_phone_number_id,
                self.meta_graph_base_url,
            )
        )

    @classmethod
    def from_env(cls) -> "WhatsAppProviderSettings":
        raw_mode = os.getenv(PROVIDER_ENV, "fake")
        if raw_mode not in {"fake", "meta_cloud"}:
            raise ValueError(
                f"{PROVIDER_ENV} must be exactly fake or meta_cloud"
            )
        return cls(
            mode=raw_mode,
            meta_access_token=_read_optional_text(META_ACCESS_TOKEN_ENV),
            meta_phone_number_id=_read_optional_text(META_PHONE_NUMBER_ID_ENV),
            meta_graph_base_url=_read_optional_text(META_GRAPH_BASE_URL_ENV),
        )

    def validate(self) -> None:
        if self.mode not in {"fake", "meta_cloud"}:
            raise ValueError(
                f"{PROVIDER_ENV} must be exactly fake or meta_cloud"
            )
        if self.mode == "fake":
            return
        missing = [
            name
            for name, value in (
                (META_ACCESS_TOKEN_ENV, self.meta_access_token),
                (META_PHONE_NUMBER_ID_ENV, self.meta_phone_number_id),
                (META_GRAPH_BASE_URL_ENV, self.meta_graph_base_url),
            )
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise ValueError(
                "missing required WhatsApp provider configuration: "
                + ", ".join(missing)
            )


@dataclass(frozen=True)
class DurableSettings:
    database_url: str | None
    redis_url: str | None
    stream_name: str = "whatsapp-support:inbound"
    consumer_group: str = "whatsapp-support-workers"
    lease_seconds: int = 60
    max_attempts: int = 3
    whatsapp: WhatsAppProviderSettings = field(
        default_factory=WhatsAppProviderSettings
    )

    @property
    def enabled(self) -> bool:
        return self.database_url is not None and self.redis_url is not None

    @classmethod
    def from_env(cls) -> "DurableSettings":
        database_url = _read_optional_url(DATABASE_URL_ENV)
        redis_url = _read_optional_url(REDIS_URL_ENV)
        if (database_url is None) != (redis_url is None):
            raise ValueError(
                f"Both {DATABASE_URL_ENV} and {REDIS_URL_ENV} must be set together"
            )
        return cls(
            database_url=database_url,
            redis_url=redis_url,
            stream_name=_read_text(
                STREAM_NAME_ENV,
                "whatsapp-support:inbound",
                legacy_name=_LEGACY_STREAM_NAME_ENV,
            ),
            consumer_group=_read_text(
                CONSUMER_GROUP_ENV,
                "whatsapp-support-workers",
                legacy_name=_LEGACY_CONSUMER_GROUP_ENV,
            ),
            lease_seconds=_read_positive_int(
                LEASE_SECONDS_ENV, 60, legacy_name=_LEGACY_LEASE_SECONDS_ENV
            ),
            max_attempts=_read_positive_int(
                MAX_ATTEMPTS_ENV, 3, legacy_name=_LEGACY_MAX_ATTEMPTS_ENV
            ),
            whatsapp=WhatsAppProviderSettings.from_env(),
        )


@dataclass(frozen=True)
class RagSettings:
    mode: Literal["fake", "openai", "compatible"] = "fake"
    index_path: str = field(default_factory=lambda: str(_default_rag_index_path()))
    openai_api_key: str | None = field(default=None, repr=False)
    embedding_model: str = "fake-embedding"
    response_model: str = "fake-model"
    embedding_dimension: int = 1
    embedding_timeout_seconds: float = 10.0
    embedding_max_retries: int = 2
    response_timeout_seconds: float = 20.0
    response_max_retries: int = 2
    vector_candidate_count: int = 20
    bm25_candidate_count: int = 20
    final_candidate_count: int = 5
    evidence_threshold: float = 0.0
    summary_trigger_turns: int = 8
    summary_retention_turns: int = 6
    reply_max_chars: int = 500
    prompt_version: str = "v1"
    embedding_api_key: str | None = field(default=None, repr=False)
    response_api_key: str | None = field(default=None, repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    response_base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> "RagSettings":
        raw_mode = os.getenv(RAG_MODE_ENV, "fake")
        if raw_mode == "compatible":
            return cls.from_provider_files(
                os.getenv("WHATSAPP_SUPPORT_EMBEDDING_CONFIG_FILE", ".rag-data/bailian.local.json"),
                os.getenv("WHATSAPP_SUPPORT_RESPONSE_CONFIG_FILE", ".rag-data/deepseek.local.json"),
            )
        if raw_mode not in {"fake", "openai"}:
            raise ValueError(f"{RAG_MODE_ENV} must be exactly fake, openai or compatible")
        mode: Literal["fake", "openai"] = raw_mode
        index_path = _read_text(
            RAG_INDEX_PATH_ENV,
            str(_default_rag_index_path()),
            legacy_name=RAG_INDEX_PATH_ENV,
        )
        if not index_path.strip():
            raise ValueError(f"{RAG_INDEX_PATH_ENV} must be a non-empty string")
        embedding_dimension = (
            _read_required_positive_int(RAG_EMBEDDING_DIMENSION_ENV)
            if mode == "openai"
            else _read_positive_int(RAG_EMBEDDING_DIMENSION_ENV, 1)
        )
        embedding_model = _read_optional_text(OPENAI_EMBEDDING_MODEL_ENV)
        response_model = _read_optional_text(OPENAI_RESPONSE_MODEL_ENV)
        openai_api_key = _read_optional_text(OPENAI_API_KEY_ENV)
        if mode == "openai":
            missing = [
                name
                for name, value in (
                    (OPENAI_API_KEY_ENV, openai_api_key),
                    (OPENAI_EMBEDDING_MODEL_ENV, embedding_model),
                    (OPENAI_RESPONSE_MODEL_ENV, response_model),
                )
                if not isinstance(value, str) or not value.strip()
            ]
            if missing:
                raise ValueError(
                    "missing required RAG OpenAI configuration: "
                    + ", ".join(missing)
                )
        settings = cls(
            mode=mode,
            index_path=index_path.strip(),
            openai_api_key=openai_api_key,
            embedding_model=embedding_model or "fake-embedding",
            response_model=response_model or "fake-model",
            embedding_dimension=embedding_dimension,
            embedding_timeout_seconds=_read_positive_float(
                RAG_EMBEDDING_TIMEOUT_ENV, 10.0
            ),
            embedding_max_retries=_read_non_negative_int(
                RAG_EMBEDDING_RETRIES_ENV, 2
            ),
            response_timeout_seconds=_read_positive_float(
                RAG_RESPONSE_TIMEOUT_ENV, 20.0
            ),
            response_max_retries=_read_non_negative_int(RAG_RESPONSE_RETRIES_ENV, 2),
            vector_candidate_count=_read_positive_int(
                RAG_VECTOR_CANDIDATES_ENV, 20
            ),
            bm25_candidate_count=_read_positive_int(RAG_BM25_CANDIDATES_ENV, 20),
            final_candidate_count=_read_positive_int(RAG_FINAL_CANDIDATES_ENV, 5),
            evidence_threshold=_read_non_negative_float(
                RAG_EVIDENCE_THRESHOLD_ENV, 0.0
            ),
            summary_trigger_turns=_read_positive_int(RAG_SUMMARY_TRIGGER_ENV, 8),
            summary_retention_turns=_read_positive_int(RAG_SUMMARY_RETENTION_ENV, 6),
            reply_max_chars=_read_positive_int(RAG_REPLY_MAX_CHARS_ENV, 500),
            prompt_version=_read_text(
                RAG_PROMPT_VERSION_ENV,
                "v1",
                legacy_name=RAG_PROMPT_VERSION_ENV,
            ).strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode not in {"fake", "openai", "compatible"}:
            raise ValueError(f"{RAG_MODE_ENV} must be exactly fake, openai or compatible")
        if not isinstance(self.index_path, str) or not self.index_path.strip():
            raise ValueError(f"{RAG_INDEX_PATH_ENV} must be a non-empty string")
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError(f"{OPENAI_EMBEDDING_MODEL_ENV} must be a non-empty string")
        if not isinstance(self.response_model, str) or not self.response_model.strip():
            raise ValueError(f"{OPENAI_RESPONSE_MODEL_ENV} must be a non-empty string")
        if not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValueError(
                f"{RAG_EMBEDDING_DIMENSION_ENV} must be a positive integer"
            )
        if (
            not math.isfinite(self.embedding_timeout_seconds)
            or self.embedding_timeout_seconds <= 0
        ):
            raise ValueError(f"{RAG_EMBEDDING_TIMEOUT_ENV} must be positive")
        if self.embedding_max_retries < 0:
            raise ValueError(
                f"{RAG_EMBEDDING_RETRIES_ENV} must be a non-negative integer"
            )
        if (
            not math.isfinite(self.response_timeout_seconds)
            or self.response_timeout_seconds <= 0
        ):
            raise ValueError(f"{RAG_RESPONSE_TIMEOUT_ENV} must be positive")
        if self.response_max_retries < 0:
            raise ValueError(
                f"{RAG_RESPONSE_RETRIES_ENV} must be a non-negative integer"
            )
        if self.vector_candidate_count <= 0:
            raise ValueError(f"{RAG_VECTOR_CANDIDATES_ENV} must be a positive integer")
        if self.bm25_candidate_count <= 0:
            raise ValueError(f"{RAG_BM25_CANDIDATES_ENV} must be a positive integer")
        if self.final_candidate_count <= 0:
            raise ValueError(f"{RAG_FINAL_CANDIDATES_ENV} must be a positive integer")
        if self.final_candidate_count > self.vector_candidate_count:
            raise ValueError(
                f"{RAG_FINAL_CANDIDATES_ENV} must be less than or equal to {RAG_VECTOR_CANDIDATES_ENV}"
            )
        if self.final_candidate_count > self.bm25_candidate_count:
            raise ValueError(
                f"{RAG_FINAL_CANDIDATES_ENV} must be less than or equal to {RAG_BM25_CANDIDATES_ENV}"
            )
        if not math.isfinite(self.evidence_threshold) or self.evidence_threshold < 0:
            raise ValueError(
                f"{RAG_EVIDENCE_THRESHOLD_ENV} must be a non-negative number"
            )
        if self.summary_trigger_turns <= self.summary_retention_turns:
            raise ValueError(
                f"{RAG_SUMMARY_TRIGGER_ENV} must be greater than {RAG_SUMMARY_RETENTION_ENV}"
            )
        if self.reply_max_chars <= 0:
            raise ValueError(f"{RAG_REPLY_MAX_CHARS_ENV} must be a positive integer")
        if not isinstance(self.prompt_version, str) or not self.prompt_version.strip():
            raise ValueError(f"{RAG_PROMPT_VERSION_ENV} must be a non-empty string")
        if self.mode == "openai" and (
            not isinstance(self.openai_api_key, str) or not self.openai_api_key.strip()
        ):
            raise ValueError(
                "missing required RAG OpenAI configuration: " + OPENAI_API_KEY_ENV
            )
        if self.mode == "compatible":
            from whatsapp_support.openai_adapters import validate_api_base_url

            validate_api_base_url(self.embedding_base_url)
            validate_api_base_url(self.response_base_url)
            for role, key in (("embedding", self.embedding_api_key), ("response", self.response_api_key)):
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"missing {role} API key in local provider configuration")

    @classmethod
    def from_provider_files(cls, embedding_path: str, response_path: str) -> "RagSettings":
        embedding = _read_provider_file(embedding_path, "embedding")
        response = _read_provider_file(response_path, "response")
        settings = cls(
            mode="compatible",
            index_path=os.getenv(RAG_INDEX_PATH_ENV, str(Path.cwd() / ".rag-index" / "bailian.sqlite3")),
            embedding_api_key=embedding["api_key"].strip(),
            response_api_key=response["api_key"].strip(),
            embedding_base_url=embedding["base_url"].strip(),
            response_base_url=response["base_url"].strip(),
            embedding_model=embedding["embedding_model"].strip(),
            response_model=response["response_model"].strip(),
            embedding_dimension=embedding["dimensions"],
            embedding_timeout_seconds=_read_positive_float(RAG_EMBEDDING_TIMEOUT_ENV, 45.0),
            embedding_max_retries=_read_non_negative_int(RAG_EMBEDDING_RETRIES_ENV, 2),
            response_timeout_seconds=_read_positive_float(RAG_RESPONSE_TIMEOUT_ENV, 90.0),
            response_max_retries=_read_non_negative_int(RAG_RESPONSE_RETRIES_ENV, 0),
            vector_candidate_count=_read_positive_int(RAG_VECTOR_CANDIDATES_ENV, 20),
            bm25_candidate_count=_read_positive_int(RAG_BM25_CANDIDATES_ENV, 20),
            final_candidate_count=_read_positive_int(RAG_FINAL_CANDIDATES_ENV, 5),
            evidence_threshold=_read_non_negative_float(RAG_EVIDENCE_THRESHOLD_ENV, 0.0),
            summary_trigger_turns=_read_positive_int(RAG_SUMMARY_TRIGGER_ENV, 8),
            summary_retention_turns=_read_positive_int(RAG_SUMMARY_RETENTION_ENV, 6),
            reply_max_chars=_read_positive_int(RAG_REPLY_MAX_CHARS_ENV, 500),
            prompt_version=os.getenv(RAG_PROMPT_VERSION_ENV, "v1"),
        )
        settings.validate()
        return settings

    def create_embedding(
        self,
        *,
        transport_factory: Callable[[], AsyncHttpTransport] | None = None,
    ):
        if self.mode == "fake":
            from whatsapp_support.fakes import FakeEmbedding

            return FakeEmbedding(
                model_name=self.embedding_model,
                dimension=self.embedding_dimension,
            )
        from whatsapp_support.meta_whatsapp import HttpxTransport
        from whatsapp_support.openai_adapters import OpenAIEmbeddingAdapter

        transport = HttpxTransport() if transport_factory is None else transport_factory()
        key = self.embedding_api_key if self.mode == "compatible" else self.openai_api_key
        assert key is not None
        return OpenAIEmbeddingAdapter(
            key,
            self.embedding_model,
            self.embedding_dimension,
            transport,
            timeout_seconds=self.embedding_timeout_seconds,
            max_retries=self.embedding_max_retries,
            base_url=self.embedding_base_url,
        )

    def create_model(
        self,
        *,
        transport_factory: Callable[[], AsyncHttpTransport] | None = None,
    ):
        if self.mode == "fake":
            from whatsapp_support.fakes import FakeModel

            return FakeModel()
        from whatsapp_support.meta_whatsapp import HttpxTransport
        from whatsapp_support.openai_adapters import OpenAIResponseAdapter

        transport = HttpxTransport() if transport_factory is None else transport_factory()
        key = self.response_api_key if self.mode == "compatible" else self.openai_api_key
        assert key is not None
        return OpenAIResponseAdapter(
            key,
            self.response_model,
            transport,
            timeout_seconds=self.response_timeout_seconds,
            max_retries=self.response_max_retries,
            base_url=self.response_base_url,
        )


def _read_provider_file(path: str, role: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        fields = ("api_key", "base_url", f"{role}_model")
        if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in fields):
            raise ValueError
        if role == "embedding" and (type(data.get("dimensions")) is not int or data["dimensions"] <= 0):
            raise ValueError
    except (OSError, ValueError, TypeError):
        raise ValueError(f"invalid {role} local provider configuration") from None
    return data


def _read_optional_url(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    trimmed = raw.strip()
    if trimmed == "":
        return None
    return trimmed


def _read_optional_text(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    trimmed = raw.strip()
    return trimmed or None


def _read_text(name: str, default: str, *, legacy_name: str) -> str:
    return os.getenv(name, os.getenv(legacy_name, default))


def _read_positive_int(
    name: str, default: int, *, legacy_name: str | None = None
) -> int:
    raw = os.getenv(name)
    if raw is None and legacy_name is not None:
        raw = os.getenv(legacy_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_required_positive_int(name: str) -> int:
    raw = os.getenv(name)
    if raw is None:
        raise ValueError(f"missing required RAG OpenAI configuration: {name}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _read_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _read_non_negative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _default_rag_index_path() -> Path:
    return Path.cwd() / ".rag-index" / "knowledge.sqlite3"
