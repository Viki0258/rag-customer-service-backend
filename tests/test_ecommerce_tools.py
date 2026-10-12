import asyncio
from datetime import UTC, datetime

from whatsapp_support.domain import ToolRequest
from whatsapp_support.ecommerce_tools import EcommerceToolRegistry, FakeEcommerceTools


NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


def make_tool_request(tool_name: str, arguments: dict[str, object]) -> ToolRequest:
    return ToolRequest(
        request_id=f"request-{tool_name}",
        conversation_id="conversation-1",
        tool_name=tool_name,
        arguments=arguments,
        trace_id="trace-1",
        requested_at=NOW,
    )


def test_fake_product_lookup_returns_stable_public_fields() -> None:
    async def scenario() -> None:
        result = await FakeEcommerceTools(clock=lambda: NOW).execute(
            make_tool_request("get_product_info", {"product_id": "sku-1"})
        )

        assert result.success is True
        assert result.output_data == {
            "product_id": "sku-1",
            "name": "Everyday Backpack",
            "price": "49.00 USD",
            "description": "A durable everyday backpack.",
        }
        assert "internal" not in result.output_data

    asyncio.run(scenario())


def test_fake_unknown_order_is_explicit_failure() -> None:
    async def scenario() -> None:
        result = await FakeEcommerceTools(clock=lambda: NOW).execute(
            make_tool_request("get_order_status", {"order_id": "missing"})
        )

        assert result.success is False
        assert result.error_code == "not_found"
        assert result.output_data == {}

    asyncio.run(scenario())


def test_fake_order_lookup_resolves_stable_verified_subject() -> None:
    async def scenario() -> None:
        result = await FakeEcommerceTools(clock=lambda: NOW).execute(
            make_tool_request("get_order_status", {"subject_id": "customer-1"})
        )

        assert result.success is True
        assert result.output_data["order_id"] == "order-1"

    asyncio.run(scenario())


def test_fake_tool_rejects_missing_or_wrong_type_arguments() -> None:
    async def scenario() -> None:
        tool = FakeEcommerceTools(clock=lambda: NOW)

        missing = await tool.execute(make_tool_request("get_product_inventory", {}))
        wrong_type = await tool.execute(
            make_tool_request("get_product_inventory", {"product_id": 123})
        )

        assert missing.success is False
        assert missing.error_code == "invalid_arguments"
        assert wrong_type.success is False
        assert wrong_type.error_code == "invalid_arguments"

    asyncio.run(scenario())


def test_registry_rejects_unregistered_tool_names() -> None:
    tool = FakeEcommerceTools(clock=lambda: NOW)
    registry = EcommerceToolRegistry({"get_product_info": tool})

    assert registry.get("get_product_info") is tool
    assert registry.get("delete_order") is None
