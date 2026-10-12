from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from whatsapp_support.domain import ToolRequest, ToolResult
from whatsapp_support.ports import BusinessToolPort


SUPPORTED_TOOL_NAMES = frozenset(
    {
        "get_product_info",
        "get_product_inventory",
        "get_order_status",
        "get_shipping_tracking",
        "check_return_eligibility",
    }
)


class EcommerceToolRegistry:
    def __init__(self, tools: Mapping[str, BusinessToolPort]) -> None:
        self._tools = {
            name: tool for name, tool in tools.items() if name in SUPPORTED_TOOL_NAMES
        }

    def get(self, tool_name: str) -> BusinessToolPort | None:
        return self._tools.get(tool_name)


class FakeEcommerceTools:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(self, request: ToolRequest) -> ToolResult:
        if request.tool_name not in SUPPORTED_TOOL_NAMES:
            return self._failure(request, "unsupported_tool")

        if request.tool_name in {"get_product_info", "get_product_inventory"}:
            key = request.arguments.get("product_id")
        else:
            key = request.arguments.get("order_id")
            if key is None:
                subject_id = request.arguments.get("subject_id")
                key = "order-1" if subject_id == "customer-1" else None
        if not isinstance(key, str) or not key.strip():
            return self._failure(request, "invalid_arguments")

        output = self._lookup(request.tool_name, key.strip())
        if output is None:
            return self._failure(request, "not_found")
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            success=True,
            output_text="ok",
            output_data=output,
            completed_at=self._clock(),
        )

    def _lookup(self, tool_name: str, key: str) -> dict[str, object] | None:
        if key == "sku-1":
            if tool_name == "get_product_info":
                return {
                    "product_id": "sku-1",
                    "name": "Everyday Backpack",
                    "price": "49.00 USD",
                    "description": "A durable everyday backpack.",
                }
            if tool_name == "get_product_inventory":
                return {
                    "product_id": "sku-1",
                    "name": "Everyday Backpack",
                    "available": True,
                    "quantity": 7,
                }
        if key == "order-1":
            if tool_name == "get_order_status":
                return {
                    "order_id": "order-1",
                    "status": "shipped",
                    "updated_at": "2026-08-27T10:00:00Z",
                }
            if tool_name == "get_shipping_tracking":
                return {
                    "order_id": "order-1",
                    "carrier": "Example Express",
                    "tracking_number": "TRACK-1",
                    "status": "in_transit",
                }
            if tool_name == "check_return_eligibility":
                return {
                    "order_id": "order-1",
                    "eligible": True,
                    "reason": "Within the 30-day return window.",
                }
        return None

    def _failure(self, request: ToolRequest, error_code: str) -> ToolResult:
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            success=False,
            output_text="",
            output_data={},
            completed_at=self._clock(),
            error_code=error_code,
        )
