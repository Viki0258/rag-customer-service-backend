from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re

from whatsapp_support.domain import (
    AutomationState,
    Conversation,
    IdentityResult,
    IdentityState,
)


READ_ONLY_INTENTS = frozenset(
    {
        "product_info",
        "product_policy",
        "order_status",
        "inventory",
        "shipping",
        "return_eligibility",
    }
)

WRITE_RISK_INTENTS = frozenset(
    {"refund", "change_address", "cancel_order", "complaint", "human"}
)

PROTECTED_SCOPES = {
    "order_status": "order:read",
    "shipping": "shipping:read",
    "return_eligibility": "return:read",
}

TOOL_BY_INTENT = {
    "product_info": "get_product_info",
    "order_status": "get_order_status",
    "inventory": "get_product_inventory",
    "shipping": "get_shipping_tracking",
    "return_eligibility": "check_return_eligibility",
}


class IntentRouter:
    """Small deterministic intent router for the fake-first phase."""

    def route(self, text: str) -> str | None:
        if not isinstance(text, str):
            return None
        normalized = text.casefold().strip()
        if not normalized:
            return None

        if any(term in normalized for term in ("refund", "\u9000\u6b3e", "\u9000\u94b1")):
            return "refund"
        if any(
            term in normalized
            for term in (
                "change address",
                "address change",
                "\u6539\u5730\u5740",
                "\u4fee\u6539\u5730\u5740",
            )
        ):
            return "change_address"
        if any(
            term in normalized
            for term in (
                "cancel order",
                "cancel my",
                "\u53d6\u6d88\u8ba2\u5355",
                "\u53d6\u6d88\u6211\u7684",
            )
        ):
            return "cancel_order"
        if any(term in normalized for term in ("complaint", "\u6295\u8bc9", "\u62b1\u6028")):
            return "complaint"
        if any(
            term in normalized
            for term in ("human", "agent", "\u4eba\u5de5", "\u5ba2\u670d", "\u771f\u4eba")
        ):
            return "human"

        if any(
            term in normalized
            for term in (
                "eligible for a return",
                "return eligibility",
                "can i return",
                "return this",
                "\u9000\u8d27\u8d44\u683c",
                "\u53ef\u4ee5\u9000\u8d27",
                "\u80fd\u9000\u8d27",
            )
        ):
            return "return_eligibility"
        if any(
            term in normalized
            for term in (
                "tracking",
                "track shipment",
                "\u7269\u6d41",
                "\u5feb\u9012",
                "\u5305\u88f9",
            )
        ):
            return "shipping"
        if any(
            term in normalized
            for term in (
                "order status",
                "where is my order",
                "\u8ba2\u5355\u72b6\u6001",
                "\u8ba2\u5355\u5230\u54ea",
                "\u8ba2\u5355",
            )
        ):
            return "order_status"
        if any(
            term in normalized
            for term in (
                "in stock",
                "stock",
                "inventory",
                "available",
                "\u5e93\u5b58",
                "\u6709\u8d27",
                "\u73b0\u8d27",
            )
        ):
            return "inventory"
        if any(
            term in normalized
            for term in (
                "return policy",
                "policy",
                "\u9000\u8d27\u653f\u7b56",
                "\u9000\u6b3e\u653f\u7b56",
                "\u5546\u54c1\u653f\u7b56",
            )
        ):
            return "product_policy"
        # General published policies are knowledge, not customer-specific tracking.
        # Protected and write-risk requests above must keep their priority.
        # Recognize the policy subject and attribute independent of word order.
        # A current explicit return topic must not inherit a prior shipping topic.
        return_topic = bool(re.search(r'\breturn(?:s|ing)?\b', normalized))
        return_attribute = bool(re.search(
            r'\b(?:conditions?|requirements?|criteria|rules?|deadlines?|time\s+limits?)\b', normalized))
        if (return_topic and return_attribute) or (
            '退货' in normalized and any(term in normalized for term in ('条件', '要求', '期限', '规则'))
        ):
            return 'product_policy'
        if any(term in normalized for term in (
            "standard shipping", "standard delivery", "return conditions",
            "标准配送", "退货条件", "退货有哪些条件",
        )):
            return "product_policy"
        if any(term in normalized for term in ("shipping", "delivery")):
            return "shipping"
        if any(
            term in normalized
            for term in (
                "product",
                "price",
                "material",
                "size",
                "\u5546\u54c1",
                "\u4ea7\u54c1",
                "\u4ef7\u683c",
                "\u6750\u8d28",
                "\u5c3a\u5bf8",
            )
        ):
            return "product_policy"
        return None


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    handoff: bool
    reason: str
    required_scope: str | None = None
    tool_name: str | None = None


class SafetyPolicy:
    def authorize(
        self,
        conversation: Conversation,
        intent: str,
        identity: IdentityResult | None = None,
        *,
        now: datetime | None = None,
    ) -> PolicyDecision:
        if conversation.automation_state == AutomationState.HUMAN_ACTIVE:
            return PolicyDecision(
                allowed=False,
                handoff=False,
                reason="human_active",
            )

        if intent in WRITE_RISK_INTENTS:
            return PolicyDecision(
                allowed=False,
                handoff=True,
                reason="write_requires_human",
            )

        required_scope = PROTECTED_SCOPES.get(intent)
        tool_name = TOOL_BY_INTENT.get(intent)
        if required_scope is not None:
            grant_time = datetime.now(UTC) if now is None else now
            if identity is not None:
                identity_allowed = (
                    identity.verified
                    and bool(identity.subject_id.strip())
                    and identity.allows(required_scope)
                    and identity.expires_at is not None
                    and grant_time < identity.expires_at
                )
            else:
                identity_allowed = (
                    conversation.identity_state == IdentityState.VERIFIED
                    and bool(conversation.identity_subject_id)
                    and required_scope in conversation.identity_scopes
                    and conversation.identity_expires_at is not None
                    and grant_time < conversation.identity_expires_at
                )
            if not identity_allowed:
                return PolicyDecision(
                    allowed=False,
                    handoff=True,
                    reason="identity_required",
                    required_scope=required_scope,
                    tool_name=tool_name,
                )

        if intent in READ_ONLY_INTENTS:
            return PolicyDecision(
                allowed=True,
                handoff=False,
                reason="allow_read_only",
                required_scope=required_scope,
                tool_name=tool_name,
            )

        return PolicyDecision(
            allowed=False,
            handoff=True,
            reason="unknown_intent",
        )
