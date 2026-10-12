from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_configured_durable_runtime_components_are_ready(
    durable_runtime_components,
) -> None:
    assert durable_runtime_components.mode == "durable"
    assert await durable_runtime_components.readyz() is True
