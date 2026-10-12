import os
from pathlib import Path

import pytest

from whatsapp_support.config import DurableSettings, RagSettings
from whatsapp_support.fakes import FakeEmbedding, FakeModel


def test_from_env_uses_fake_first_defaults_when_urls_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHATSAPP_SUPPORT_DATABASE_URL", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_URL", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_STREAM", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_GROUP", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS", raising=False)

    settings = DurableSettings.from_env()

    assert settings.database_url is None
    assert settings.redis_url is None
    assert settings.stream_name == "whatsapp-support:inbound"
    assert settings.consumer_group == "whatsapp-support-workers"
    assert settings.lease_seconds == 60
    assert settings.max_attempts == 3
    assert settings.enabled is False


def test_provider_defaults_to_fake_without_meta_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "WHATSAPP_SUPPORT_PROVIDER",
        "WHATSAPP_SUPPORT_META_ACCESS_TOKEN",
        "WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID",
        "WHATSAPP_SUPPORT_META_GRAPH_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = DurableSettings.from_env()

    assert settings.whatsapp.mode == "fake"
    assert settings.whatsapp.validate() is None
    assert settings.whatsapp.is_complete is False


def test_provider_rejects_unknown_mode_without_echoing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_PROVIDER", "unexpected")

    with pytest.raises(ValueError, match="WHATSAPP_SUPPORT_PROVIDER"):
        DurableSettings.from_env()


def test_meta_provider_validation_names_missing_variables_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_PROVIDER", "meta_cloud")
    monkeypatch.setenv(
        "WHATSAPP_SUPPORT_META_ACCESS_TOKEN", "super-secret-token"
    )
    monkeypatch.delenv(
        "WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID", raising=False
    )
    monkeypatch.delenv(
        "WHATSAPP_SUPPORT_META_GRAPH_BASE_URL", raising=False
    )

    settings = DurableSettings.from_env()

    with pytest.raises(ValueError) as excinfo:
        settings.whatsapp.validate()

    error = str(excinfo.value)
    assert "WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID" in error
    assert "WHATSAPP_SUPPORT_META_GRAPH_BASE_URL" in error
    assert "super-secret-token" not in error


def test_from_env_enables_durable_mode_only_when_both_urls_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_DATABASE_URL", "postgresql://user:secret@db/app")
    monkeypatch.setenv("WHATSAPP_SUPPORT_REDIS_URL", "redis://:secret@cache/0")
    monkeypatch.setenv("WHATSAPP_SUPPORT_REDIS_STREAM", "custom-stream")
    monkeypatch.setenv("WHATSAPP_SUPPORT_REDIS_GROUP", "custom-group")
    monkeypatch.setenv("WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS", "45")
    monkeypatch.setenv("WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS", "5")

    settings = DurableSettings.from_env()

    assert settings.database_url == "postgresql://user:secret@db/app"
    assert settings.redis_url == "redis://:secret@cache/0"
    assert settings.stream_name == "custom-stream"
    assert settings.consumer_group == "custom-group"
    assert settings.lease_seconds == 45
    assert settings.max_attempts == 5
    assert settings.enabled is True


def test_from_env_rejects_one_sided_url_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_DATABASE_URL", "postgresql://user:secret@db/app")
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_URL", raising=False)

    with pytest.raises(ValueError, match="Both WHATSAPP_SUPPORT_DATABASE_URL and WHATSAPP_SUPPORT_REDIS_URL must be set together"):
        DurableSettings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS", "0"),
        ("WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS", "-1"),
        ("WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS", "abc"),
    ],
)
def test_from_env_rejects_non_positive_integer_settings(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.delenv("WHATSAPP_SUPPORT_DATABASE_URL", raising=False)
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_URL", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        DurableSettings.from_env()


def test_from_env_does_not_log_secret_values_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_url = "postgresql://user:super-secret@db/app"
    monkeypatch.setenv("WHATSAPP_SUPPORT_DATABASE_URL", secret_url)
    monkeypatch.delenv("WHATSAPP_SUPPORT_REDIS_URL", raising=False)

    with pytest.raises(ValueError) as excinfo:
        DurableSettings.from_env()

    assert secret_url not in str(excinfo.value)


def test_rag_settings_default_to_fake_mode_and_project_local_index(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WHATSAPP_SUPPORT_RAG_MODE",
        "WHATSAPP_SUPPORT_RAG_INDEX_PATH",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_RESPONSE_MODEL",
        "WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION",
        "WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS",
        "WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES",
        "WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS",
        "WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES",
        "WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES",
        "WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES",
        "WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES",
        "WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD",
        "WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS",
        "WHATSAPP_SUPPORT_RAG_SUMMARY_RETENTION_TURNS",
        "WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RagSettings.from_env()

    assert settings.mode == "fake"
    assert settings.openai_api_key is None
    assert Path(settings.index_path).name == "knowledge.sqlite3"
    assert Path(settings.index_path).parent.name == ".rag-index"
    assert settings.embedding_model == "fake-embedding"
    assert settings.response_model == "fake-model"
    assert settings.embedding_dimension == 1
    assert settings.vector_candidate_count == 20
    assert settings.bm25_candidate_count == 20
    assert settings.final_candidate_count == 5
    assert settings.evidence_threshold == 0.0
    assert settings.summary_trigger_turns == 8
    assert settings.summary_retention_turns == 6
    assert settings.reply_max_chars == 500


def test_rag_settings_openai_mode_reads_independent_model_and_retry_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_MODE", "openai")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_INDEX_PATH", "custom-index.sqlite3")
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_RESPONSE_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION", "3072")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES", "1")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS", "12.0")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES", "3")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES", "25")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES", "21")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES", "4")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD", "0.02")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS", "10")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_RETENTION_TURNS", "6")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS", "600")

    settings = RagSettings.from_env()

    assert settings.mode == "openai"
    assert settings.index_path == "custom-index.sqlite3"
    assert settings.openai_api_key == "super-secret-key"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.response_model == "gpt-5.6-sol"
    assert settings.embedding_dimension == 3072
    assert settings.embedding_timeout_seconds == 4.5
    assert settings.embedding_max_retries == 1
    assert settings.response_timeout_seconds == 12.0
    assert settings.response_max_retries == 3
    assert settings.vector_candidate_count == 25
    assert settings.bm25_candidate_count == 21
    assert settings.final_candidate_count == 4
    assert settings.evidence_threshold == 0.02
    assert settings.summary_trigger_turns == 10
    assert settings.summary_retention_turns == 6
    assert settings.reply_max_chars == 600
    assert "super-secret-key" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("WHATSAPP_SUPPORT_RAG_MODE", "bogus", "WHATSAPP_SUPPORT_RAG_MODE"),
        ("WHATSAPP_SUPPORT_RAG_INDEX_PATH", "   ", "WHATSAPP_SUPPORT_RAG_INDEX_PATH"),
        (
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION",
            "0",
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS",
            "0",
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES",
            "-1",
            "WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS",
            "-1",
            "WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES",
            "-1",
            "WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES",
            "0",
            "WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES",
            "0",
            "WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES",
            "0",
            "WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD",
            "-0.01",
            "WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD",
        ),
        *[
            (name, value, name)
            for name in (
                "WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS",
                "WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS",
                "WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD",
            )
            for value in ("nan", "inf", "-inf")
        ],
        (
            "WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS",
            "6",
            "WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS",
        ),
        (
            "WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS",
            "0",
            "WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS",
        ),
    ],
)
def test_rag_settings_reject_invalid_values_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    match: str,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_MODE", "openai")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_INDEX_PATH", "knowledge.sqlite3")
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_RESPONSE_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION", "3072")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_MAX_RETRIES", "1")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_RESPONSE_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_RESPONSE_MAX_RETRIES", "2")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_VECTOR_CANDIDATES", "20")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_BM25_CANDIDATES", "20")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_FINAL_CANDIDATES", "4")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EVIDENCE_THRESHOLD", "0.0")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS", "8")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_RETENTION_TURNS", "6")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS", "500")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match) as excinfo:
        RagSettings.from_env()

    assert "super-secret-key" not in str(excinfo.value)


def test_rag_settings_reject_openai_mode_when_models_or_key_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_MODE", "openai")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_INDEX_PATH", "knowledge.sqlite3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_RESPONSE_MODEL", raising=False)
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION", "3072")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_TRIGGER_TURNS", "8")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_SUMMARY_RETENTION_TURNS", "6")
    monkeypatch.setenv("WHATSAPP_SUPPORT_RAG_REPLY_MAX_CHARS", "500")

    with pytest.raises(ValueError) as excinfo:
        RagSettings.from_env()

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "OPENAI_EMBEDDING_MODEL" in message
    assert "OPENAI_RESPONSE_MODEL" in message


def test_rag_settings_fake_mode_selects_fakes_before_any_transport_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHATSAPP_SUPPORT_RAG_MODE", raising=False)
    settings = RagSettings.from_env()

    def fail_transport_factory() -> object:
        raise AssertionError("fake mode must not construct OpenAI transport")

    embedding = settings.create_embedding(transport_factory=fail_transport_factory)
    model = settings.create_model(transport_factory=fail_transport_factory)

    assert isinstance(embedding, FakeEmbedding)
    assert isinstance(model, FakeModel)
