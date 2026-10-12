from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
import pytest_asyncio

from whatsapp_support.config import (
    DATABASE_URL_ENV,
    REDIS_URL_ENV,
    DurableSettings,
)
from whatsapp_support.runtime import RuntimeComponents, create_runtime_components_async


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: Sequence[pytest.Item],
) -> None:
    del config
    if _configured(DATABASE_URL_ENV) or _configured(REDIS_URL_ENV):
        return
    skip = pytest.mark.skip(
        reason=(
            "integration dependencies are not configured: set "
            f"{DATABASE_URL_ENV} and {REDIS_URL_ENV}"
        )
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest_asyncio.fixture
async def durable_runtime_components() -> RuntimeComponents:
    try:
        settings = DurableSettings.from_env()
    except ValueError:
        pytest.skip(
            "durable runtime integration requires both "
            f"{DATABASE_URL_ENV} and {REDIS_URL_ENV}"
        )
    if not settings.enabled:
        pytest.skip(
            "integration dependencies are not configured: set "
            f"{DATABASE_URL_ENV} and {REDIS_URL_ENV}"
        )
    components = await create_runtime_components_async(settings)
    try:
        yield components
    finally:
        await components.aclose()


def _configured(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip() != ""
