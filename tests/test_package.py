import tomllib
from pathlib import Path


def test_package_exposes_version():
    import whatsapp_support

    assert whatsapp_support.__version__ == "0.1.0"


def test_test_extra_installs_redis_for_durable_entrypoint_tests() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    test_dependencies = pyproject["project"]["optional-dependencies"]["test"]

    assert "redis>=5.0" in test_dependencies


def test_durable_runtime_docs_cover_fake_first_and_optional_integration_checks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    local_development = (repo_root / "docs" / "local-development.md").read_text(
        encoding="utf-8"
    )

    combined_docs = f"{readme}\n{local_development}"

    required_phrases = [
        "fake-first",
        "postgresql+asyncpg",
        "redis://",
        "WHATSAPP_SUPPORT_DATABASE_URL",
        "WHATSAPP_SUPPORT_REDIS_URL",
        "WHATSAPP_SUPPORT_REDIS_STREAM",
        "WHATSAPP_SUPPORT_REDIS_GROUP",
        "WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS",
        "WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS",
        "docker compose up -d postgres redis",
        "pytest -m integration",
        "integration-check",
        "git diff --check",
        "lease recovery",
        "stable dedup",
        "/readyz",
        "provider timeout ambiguity",
        ".[test]",
    ]

    for phrase in required_phrases:
        assert phrase in combined_docs

    assert (
        "postgresql+asyncpg://whatsapp_support:whatsapp_support@127.0.0.1:5432/whatsapp_support"
        not in combined_docs
    )
    assert "redis://127.0.0.1:6379/0" not in combined_docs
    assert "<set this securely>" in combined_docs
    assert "<postgresql+asyncpg URL configured outside logs>" in combined_docs
    assert "<redis:// URL configured outside logs>" in combined_docs
    assert "python -m pytest" not in combined_docs


def test_docs_describe_conversation_orchestration_v1_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    combined_docs = "\n".join(
        [
            (repo_root / "README.md").read_text(encoding="utf-8"),
            (repo_root / "docs" / "local-development.md").read_text(
                encoding="utf-8"
            ),
        ]
    )

    for phrase in (
        "conversation orchestration",
        "get_product_info",
        "get_product_inventory",
        "order:read",
        "shipping:read",
        "return:read",
        "write-risk",
        "deterministic",
        "live external integration is not exercised or claimed",
    ):
        assert phrase in combined_docs


def test_docs_describe_rag_local_development_and_safety_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rag_guide = repo_root / "docs" / "rag-local-development.md"
    combined_docs = "\n".join(
        [
            (repo_root / "README.md").read_text(encoding="utf-8"),
            (repo_root / "docs" / "local-development.md").read_text(
                encoding="utf-8"
            ),
            rag_guide.read_text(encoding="utf-8"),
        ]
    )

    for phrase in (
        "py -3.12 -m whatsapp_support rag-index knowledge/",
        "WHATSAPP_SUPPORT_RAG_MODE",
        "WHATSAPP_SUPPORT_RAG_INDEX_PATH",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_RESPONSE_MODEL",
        ".rag-data/",
        ".rag-index/",
        "same API key",
        "separately configured",
        "six complete turns",
        "full response",
        "citation-ID membership",
        "semantic factuality",
        "human handoff",
        "evidence is not exposed to customers",
        "order, shipping, inventory, or return eligibility",
        "one-time indexing",
        "worker-time",
        "fake",
        "openai",
    ):
        assert phrase.casefold() in combined_docs.casefold()

    assert "recommended model" not in combined_docs.lower()


def test_readme_describes_available_adapters_without_claiming_live_verification() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "PostgreSQL, Redis, OpenAI embedding, and OpenAI response adapters exist" in readme
    assert "live external integration is not exercised or claimed" in readme
    assert "Remaining live integrations include:" not in readme
    assert "durable Inbox/Conversation/Outbox storage" not in readme
