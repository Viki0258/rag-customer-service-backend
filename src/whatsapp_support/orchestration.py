from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from whatsapp_support.domain import Conversation, EventType, InboundEvent
from whatsapp_support.ecommerce_tools import EcommerceToolRegistry, SUPPORTED_TOOL_NAMES
from whatsapp_support.policy import PROTECTED_SCOPES, IntentRouter, SafetyPolicy
from whatsapp_support.ports import BusinessToolPort, IdentityPort
from whatsapp_support.replies import ReplyComposer, ReplyDraft


@dataclass(frozen=True)
class OrchestrationResult:
    status: str
    intent: str | None
    reply: ReplyDraft | None
    trace_id: str
    handoff_reason: str | None = None
    required_scope: str | None = None
    tool_name: str | None = None
    reason: str | None = None


class ConversationOrchestrator:
    def __init__(
        self,
        tools: BusinessToolPort | None,
        identity: IdentityPort | None,
        router: IntentRouter,
        policy: SafetyPolicy,
        composer: ReplyComposer,
        *,
        registry: EcommerceToolRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
        locale: str = "en",
        memory=None,
    ) -> None:
        self._identity = identity
        self._router = router
        self._policy = policy
        self._composer = composer
        self.registry = registry or EcommerceToolRegistry(
            {} if tools is None else {name: tools for name in SUPPORTED_TOOL_NAMES}
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._locale = locale
        self._memory = memory

    async def resolve_followup_intent(self, event):
        """Resolve bounded read-only repeat requests, never infer write authority."""
        if self._memory is None or not re.fullmatch(
                r'\s*(and now|check again|what about now|again|再查一下|现在呢)[?？.!。\s]*',
                event.text or '', re.I):
            return None
        turns = await self._memory.get_recent_turns(event.conversation_id, limit=32)
        for turn in reversed(turns):
            intent = IntentRouter().route(turn.customer_text)
            if intent:
                return intent if intent in {'inventory', 'order_status', 'shipping', 'return_eligibility'} else None
        return None

    async def run(
        self,
        event: InboundEvent,
        conversation: Conversation,
    ) -> OrchestrationResult:
        trace_id = _trace_id(event.event_key)
        if event.event_type is not EventType.MESSAGE:
            return OrchestrationResult(
                status="ignored",
                intent=None,
                reply=None,
                trace_id=trace_id,
                reason="non_message_event",
            )
        if conversation.automation_state.value == "HUMAN_ACTIVE":
            return OrchestrationResult(
                status="ignored",
                intent=None,
                reply=None,
                trace_id=trace_id,
                reason="human_active",
            )
        if event.text is None or not event.text.strip():
            return self._handoff(trace_id, None, "missing_message_text")

        intent = self._router.route(event.text)
        if intent is None:
            try:
                intent = await self.resolve_followup_intent(event)
            except Exception:
                return self._handoff(trace_id, None, 'tool_context_unresolved')
        if intent is None:
            return self._handoff(trace_id, None, "unknown_intent")

        now = self._clock()
        identity_result = None
        decision = self._policy.authorize(conversation, intent, now=now)
        if not decision.allowed:
            required_scope = PROTECTED_SCOPES.get(intent)
            if required_scope is None:
                return self._handoff(trace_id, intent, decision.reason)
            if self._identity is None or event.identity_proof is None:
                return self._handoff(
                    trace_id,
                    intent,
                    "identity_required",
                    required_scope=required_scope,
                    tool_name=decision.tool_name,
                )
            try:
                identity_result = await self._identity.verify(
                    conversation.conversation_id, event.identity_proof
                )
            except Exception:
                return self._handoff(
                    trace_id,
                    intent,
                    "identity_verification_failed",
                    required_scope=required_scope,
                    tool_name=decision.tool_name,
                )
            decision = self._policy.authorize(
                conversation, intent, identity_result, now=now
            )
            if not decision.allowed:
                return self._handoff(
                    trace_id,
                    intent,
                    "identity_not_authorized",
                    required_scope=required_scope,
                    tool_name=decision.tool_name,
                )

        tool_name = decision.tool_name
        tool = self.registry.get(tool_name) if self.registry and tool_name else None
        if tool is None:
            return self._handoff(trace_id, intent, "tool_unavailable", tool_name=tool_name)
        subject_id = (
            identity_result.subject_id
            if identity_result is not None
            else conversation.identity_subject_id
        )
        argument_text = event.text
        try:
            if self._memory is not None:
                # More raw turns may be inspected locally for exact identifiers;
                # they are not appended to an LLM prompt or treated as facts.
                turns = await self._memory.get_recent_turns(event.conversation_id, limit=32)
                if self._router.route(event.text) is None:
                    argument_text = 'that ' + event.text
                argument_text = resolve_tool_reference(intent, argument_text, turns)
            arguments = _tool_arguments(intent, argument_text, subject_id)
        except Exception:
            return self._handoff(trace_id, intent, 'tool_context_unresolved', tool_name=tool_name)
        if arguments is None:
            return self._handoff(trace_id, intent, "tool_arguments_missing", tool_name=tool_name)
        try:
            result = await tool.execute(
                _tool_request(event, tool_name, arguments, trace_id, now)
            )
        except Exception:
            return self._handoff(trace_id, intent, "tool_failed", tool_name=tool_name)
        if not result.success:
            return self._handoff(trace_id, intent, "tool_failed", tool_name=tool_name)
        for key in ('product_id', 'order_id', 'subject_id'):
            if key in arguments and key in result.output_data and arguments[key] != result.output_data[key]:
                return self._handoff(trace_id, intent, 'tool_target_mismatch', tool_name=tool_name)
        reply = self._composer.compose(intent, result.output_data, self._locale)
        if reply is None:
            return self._handoff(
                trace_id, intent, "tool_result_invalid", tool_name=tool_name
            )
        return OrchestrationResult(
            status="reply",
            intent=intent,
            reply=reply,
            trace_id=trace_id,
            required_scope=decision.required_scope,
            tool_name=tool_name,
            reason=decision.reason,
        )

    @staticmethod
    def _handoff(
        trace_id: str,
        intent: str | None,
        reason: str,
        *,
        required_scope: str | None = None,
        tool_name: str | None = None,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            status="handoff",
            intent=intent,
            reply=None,
            trace_id=trace_id,
            handoff_reason=reason,
            required_scope=required_scope,
            tool_name=tool_name,
            reason=reason,
        )


def _trace_id(event_key: str) -> str:
    digest = hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:20]
    return f"trace-{digest}"


def _tool_arguments(
    intent: str, text: str, subject_id: str | None = None
) -> dict[str, object] | None:
    if intent in {"product_info", "inventory"}:
        match = re.search(r"\bsku-[a-z0-9-]+\b", text.casefold())
        return {"product_id": match.group(0)} if match else None
    if intent in {"order_status", "shipping", "return_eligibility"}:
        match = re.search(r"\border-[a-z0-9-]+\b", text.casefold())
        arguments: dict[str, object] = {}
        if match:
            arguments["order_id"] = match.group(0)
        if subject_id is not None and subject_id.strip():
            arguments["subject_id"] = subject_id.strip()
        return arguments or None
    return None


def resolve_tool_reference(intent, text, turns):
    """Reuse only explicit customer identifiers, never assistant claims or summary.

    Memory supplies a reference, not ownership. Caller must authorize first and
    the business backend must verify subject ownership of the supplied order.
    """
    prefix = 'sku' if intent in {'inventory', 'product_info'} else 'order'
    pattern = rf'\b{prefix}-[a-z0-9-]+\b'
    current = set(re.findall(pattern, text.casefold()))
    if len(current) > 1:
        raise ValueError('Ambiguous current target')
    if current or not re.search(r'\b(it|that|same|this|its)\b|这个|那个|刚才|同一', text, re.I):
        return text
    for turn in reversed(turns):
        customer = turn.customer_text
        ids = set(re.findall(pattern, customer.casefold()))
        if len(ids) > 1:
            raise ValueError('Ambiguous previous targets')
        if ids:
            return text + ' ' + next(iter(ids))
        # Do not reach across an explicit unrelated topic or protected request.
        prior_intent = IntentRouter().route(customer)
        family = {'inventory', 'product_info'} if prefix == 'sku' else {'order_status', 'shipping', 'return_eligibility'}
        if prior_intent and prior_intent not in family:
            break
    raise ValueError('No unambiguous customer target')


def _tool_request(event, tool_name, arguments, trace_id, requested_at):
    from whatsapp_support.domain import ToolRequest

    return ToolRequest(
        request_id=f"tool:{trace_id}",
        conversation_id=event.conversation_id,
        tool_name=tool_name,
        arguments=arguments,
        trace_id=trace_id,
        requested_at=requested_at,
    )
