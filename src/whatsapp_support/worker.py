from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Callable

from whatsapp_support.domain import (
    AutomationState,
    Conversation,
    EventType,
    HandoffRequest,
    IdentityResult,
    IdentityState,
    InboundEvent,
    OutboundCommand,
    OutboundMode,
    RagAnswerResult,
    window_mode,
)
from whatsapp_support.orchestration import ConversationOrchestrator
from whatsapp_support.policy import (
    PROTECTED_SCOPES,
    TOOL_BY_INTENT,
    IntentRouter,
    PolicyDecision,
    SafetyPolicy,
)
from whatsapp_support.ports import (
    BusinessToolPort,
    ConversationOutboxTransactionPort,
    ConversationPort,
    HandoffPort,
    IdentityPort,
    KnowledgePort,
    ModelPort,
    OutboxPort,
    RagAnswerPort,
    FollowupRoutingPort,
    TemplateCatalogPort,
    WhatsAppPort,
)
from whatsapp_support.replies import ReplyComposer


class RetryableProviderError(ConnectionError):
    """Explicit marker for a transient outbound provider failure."""


class PermanentProviderError(RuntimeError):
    """Explicit marker for a non-retryable outbound provider failure."""


@dataclass(frozen=True)
class WorkerResult:
    status: str
    intent: str | None
    outbox_created: bool
    handoff_created: bool
    trace_id: str
    reason: str | None = None
    retryable: bool = False
    retrieved_chunk_ids: tuple[str, ...] = ()
    supported_chunk_ids: tuple[str, ...] = ()
    prompt_version: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class DispatchResult:
    status: str
    dedup_key: str | None = None
    provider_message_id: str | None = None
    attempts: int = 0
    error: str | None = None


class ConversationWorker:
    def __init__(
        self,
        conversations: ConversationPort,
        outbox: OutboxPort,
        tools: BusinessToolPort | None,
        identity: IdentityPort | None,
        knowledge: KnowledgePort | None,
        model: ModelPort | None,
        templates: TemplateCatalogPort | None,
        handoff: HandoffPort | None,
        *,
        router: IntentRouter | None = None,
        policy: SafetyPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        locale: str = "en",
        template_name: str = "customer_service_reply",
        transaction: ConversationOutboxTransactionPort | None = None,
        orchestrator: ConversationOrchestrator | None = None,
        rag_answer: RagAnswerPort | None = None,
    ) -> None:
        self._conversations = conversations
        self._outbox = outbox
        self._identity = identity
        del knowledge, model
        self._templates = templates
        self._handoff = handoff
        self._router = router or IntentRouter()
        self._policy = policy or SafetyPolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._locale = locale
        self._template_name = template_name
        self._transaction = transaction
        self._rag_answer = rag_answer
        self._orchestrator = orchestrator or ConversationOrchestrator(
            tools=tools,
            identity=identity,
            router=self._router,
            policy=self._policy,
            composer=ReplyComposer(),
            clock=self._clock,
            locale=locale,
        )

    async def process(self, event: InboundEvent) -> WorkerResult:
        if self._transaction is not None:
            return await self._process_durable(event)

        trace_id = _trace_id(event.event_key)
        if event.event_type is not EventType.MESSAGE:
            return self._result(
                status="ignored",
                intent=None,
                trace_id=trace_id,
                reason="non_message_event",
            )

        conversation, changed = await self._apply_event(event)
        if not changed:
            return self._result(
                status="stale_or_duplicate",
                intent=None,
                trace_id=trace_id,
                reason="conversation_version_unchanged",
            )
        if conversation.automation_state == AutomationState.HUMAN_ACTIVE:
            return self._result(
                status="human_active",
                intent=None,
                trace_id=trace_id,
                reason="human_active",
            )
        if not event.text or not event.text.strip():
            return await self._handoff_result(
                conversation, None, trace_id, "missing_message_text"
            )

        intent = await self._route_with_followup(event)
        if intent is None:
            return await self._handoff_result(
                conversation, None, trace_id, "unknown_intent"
            )

        identity_result: IdentityResult | None = None
        required_scope = PROTECTED_SCOPES.get(intent)
        grant_time = self._now()
        if (
            required_scope is not None
            and not self._policy.authorize(
                conversation, intent, now=grant_time
            ).allowed
        ):
            if self._identity is None or event.identity_proof is None:
                return await self._handoff_result(
                    conversation, intent, trace_id, "identity_required"
                )
            try:
                identity_result = await self._identity.verify(
                    conversation.conversation_id, event.identity_proof
                )
            except Exception:
                return await self._handoff_result(
                    conversation, intent, trace_id, "identity_verification_failed"
                )
            if not self._policy.authorize(
                conversation, intent, identity_result, now=grant_time
            ).allowed:
                return await self._handoff_result(
                    conversation, intent, trace_id, "identity_not_authorized"
                )
            conversation = replace(
                conversation,
                identity_state=IdentityState.VERIFIED,
                identity_subject_id=identity_result.subject_id,
                identity_scopes=identity_result.scopes,
                identity_expires_at=identity_result.expires_at,
                version=conversation.version + 1,
            )
            await self._conversations.save(conversation)

        decision = self._policy.authorize(
            conversation, intent, identity_result, now=grant_time
        )
        if not decision.allowed:
            if decision.handoff:
                return await self._handoff_result(
                    conversation, intent, trace_id, decision.reason
                )
            return self._result(
                status="blocked",
                intent=intent,
                trace_id=trace_id,
                reason=decision.reason,
            )

        rag_result: RagAnswerResult | None = None
        if intent in {"product_info", "product_policy"}:
            if self._rag_answer is None:
                return await self._handoff_result(
                    conversation, intent, trace_id, "rag_not_configured"
                )
            rag_result = await self._rag_answer.answer(event, self._locale, trace_id)
            if rag_result.status == 'clarify':
                # Legacy path cannot atomically persist the clarification turn.
                return await self._handoff_result(
                    conversation, intent, trace_id, 'clarification_requires_transaction',
                    rag_result=rag_result,
                )
            if rag_result.status not in {"reply", "clarify"} or rag_result.reply is None:
                return await self._handoff_result(
                    conversation,
                    intent,
                    trace_id,
                    rag_result.reason or "insufficient_evidence",
                    rag_result=rag_result,
                )
            reply = rag_result.reply
        elif intent in TOOL_BY_INTENT:
            orchestration = await self._orchestrator.run(event, conversation)
            if orchestration.status == "handoff":
                return await self._handoff_result(
                    conversation,
                    orchestration.intent,
                    orchestration.trace_id,
                    orchestration.handoff_reason or "orchestration_handoff",
                )
            if orchestration.status != "reply" or orchestration.reply is None:
                return self._result(
                    status=orchestration.status,
                    intent=orchestration.intent,
                    trace_id=orchestration.trace_id,
                    reason=orchestration.reason,
                )
            reply = orchestration.reply.text
            trace_id = orchestration.trace_id
        else:
            return await self._handoff_result(
                conversation, intent, trace_id, "tool_unavailable_or_invalid"
            )

        return await self._enqueue_reply(
            conversation, intent, decision, reply, event, trace_id, rag_result
        )

    async def _process_durable(self, event: InboundEvent) -> WorkerResult:
        trace_id = _trace_id(event.event_key)
        if event.event_type is not EventType.MESSAGE:
            return self._result(
                status="ignored",
                intent=None,
                trace_id=trace_id,
                reason="non_message_event",
            )

        conversation, base_version, changed = await self._stage_event(event)
        if not changed:
            return await self._commit_durable(
                event=event,
                expected_version=base_version,
                conversation=conversation,
                command=None,
                status="stale_or_duplicate",
                intent=None,
                trace_id=trace_id,
                reason="conversation_version_unchanged",
            )
        if conversation.automation_state == AutomationState.HUMAN_ACTIVE:
            return await self._commit_durable(
                event=event,
                expected_version=base_version,
                conversation=conversation,
                command=None,
                status="human_active",
                intent=None,
                trace_id=trace_id,
                reason="human_active",
            )
        if not event.text or not event.text.strip():
            return await self._handoff_result_durable(
                event, base_version, conversation, None, trace_id, "missing_message_text"
            )

        intent = await self._route_with_followup(event)
        if intent is None:
            return await self._handoff_result_durable(
                event, base_version, conversation, None, trace_id, "unknown_intent"
            )

        identity_result: IdentityResult | None = None
        required_scope = PROTECTED_SCOPES.get(intent)
        grant_time = self._now()
        if (
            required_scope is not None
            and not self._policy.authorize(
                conversation, intent, now=grant_time
            ).allowed
        ):
            if self._identity is None or event.identity_proof is None:
                return await self._handoff_result_durable(
                    event, base_version, conversation, intent, trace_id, "identity_required"
                )
            try:
                identity_result = await self._identity.verify(
                    conversation.conversation_id, event.identity_proof
                )
            except Exception:
                return await self._handoff_result_durable(
                    event,
                    base_version,
                    conversation,
                    intent,
                    trace_id,
                    "identity_verification_failed",
                )
            if not self._policy.authorize(
                conversation, intent, identity_result, now=grant_time
            ).allowed:
                return await self._handoff_result_durable(
                    event, base_version, conversation, intent, trace_id, "identity_not_authorized"
                )
            conversation = replace(
                conversation,
                identity_state=IdentityState.VERIFIED,
                identity_subject_id=identity_result.subject_id,
                identity_scopes=identity_result.scopes,
                identity_expires_at=identity_result.expires_at,
                version=conversation.version + 1,
            )

        decision = self._policy.authorize(
            conversation, intent, identity_result, now=grant_time
        )
        if not decision.allowed:
            if decision.handoff:
                return await self._handoff_result_durable(
                    event, base_version, conversation, intent, trace_id, decision.reason
                )
            return await self._commit_durable(
                event=event,
                expected_version=base_version,
                conversation=conversation,
                command=None,
                status="blocked",
                intent=intent,
                trace_id=trace_id,
                reason=decision.reason,
            )

        rag_result: RagAnswerResult | None = None
        if intent in {"product_info", "product_policy"}:
            if self._rag_answer is None:
                return await self._handoff_result_durable(
                    event,
                    base_version,
                    conversation,
                    intent,
                    trace_id,
                    "rag_not_configured",
                )
            rag_result = await self._rag_answer.answer(event, self._locale, trace_id)
            if rag_result.status not in {"reply", "clarify"} or rag_result.reply is None:
                return await self._handoff_result_durable(
                    event,
                    base_version,
                    conversation,
                    intent,
                    trace_id,
                    rag_result.reason or "insufficient_evidence",
                    rag_result=rag_result,
                )
            reply = rag_result.reply
        elif intent in TOOL_BY_INTENT:
            orchestration = await self._orchestrator.run(event, conversation)
            if orchestration.status == "handoff":
                return await self._handoff_result_durable(
                    event,
                    base_version,
                    conversation,
                    orchestration.intent,
                    orchestration.trace_id,
                    orchestration.handoff_reason or "orchestration_handoff",
                )
            if orchestration.status != "reply" or orchestration.reply is None:
                return await self._commit_durable(
                    event=event,
                    expected_version=base_version,
                    conversation=conversation,
                    command=None,
                    status=orchestration.status,
                    intent=orchestration.intent,
                    trace_id=orchestration.trace_id,
                    reason=orchestration.reason,
                )
            reply = orchestration.reply.text
            trace_id = orchestration.trace_id
        else:
            return await self._handoff_result_durable(
                event,
                base_version,
                conversation,
                intent,
                trace_id,
                "tool_unavailable_or_invalid",
            )

        return await self._enqueue_reply_durable(
            event,
            base_version,
            conversation,
            intent,
            decision,
            reply,
            trace_id,
            rag_result,
        )

    async def _route_with_followup(self, event: InboundEvent) -> str | None:
        intent = self._router.route(event.text)
        if intent is not None:
            return intent
        resolver = getattr(self._orchestrator, 'resolve_followup_intent', None)
        if resolver is not None:
            try:
                resolved = await resolver(event)
            except Exception:
                return None
            # Policy and identity checks below still run for every repeated read.
            if resolved in {'inventory', 'order_status', 'shipping', 'return_eligibility'}:
                return resolved
        if isinstance(self._rag_answer, FollowupRoutingPort):
            try:
                resolved = await self._rag_answer.resolve_followup_intent(event)
            except Exception:
                return None
            # The fallback may authorize only a public knowledge query, never a tool.
            if resolved == 'product_policy':
                return resolved
        return None

    async def _stage_event(
        self, event: InboundEvent
    ) -> tuple[Conversation, int, bool]:
        current = await self._conversations.get(event.conversation_id)
        if current is None:
            current = Conversation(
                conversation_id=event.conversation_id,
                contact_id=event.contact_id,
                phone_number_id=event.phone_number_id,
            )
        expected_version = current.version
        conversation, changed = current.apply_inbound(event)
        return conversation, expected_version, changed

    async def _apply_event(self, event: InboundEvent) -> tuple[Conversation, bool]:
        updater = getattr(self._conversations, "update_from_event", None)
        if updater is not None:
            return await updater(event)

        current = await self._conversations.get(event.conversation_id)
        if current is None:
            current = Conversation(
                conversation_id=event.conversation_id,
                contact_id=event.contact_id,
                phone_number_id=event.phone_number_id,
            )
        updated, changed = current.apply_inbound(event)
        if changed:
            await self._conversations.save(updated)
        return updated, changed

    async def _enqueue_reply(
        self,
        conversation: Conversation,
        intent: str,
        decision: PolicyDecision,
        reply: str,
        event: InboundEvent,
        trace_id: str,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        mode = window_mode(conversation, self._now())
        if mode.value == "blocked":
            return self._result(
                status="blocked",
                intent=intent,
                trace_id=trace_id,
                reason="human_active",
                rag_result=rag_result,
            )

        if mode.value == "template":
            if self._templates is None:
                return await self._handoff_result(
                    conversation, intent, trace_id, "template_required"
                )
            try:
                template = await self._templates.resolve(
                    self._template_name, self._locale
                )
            except Exception:
                template = None
            if template is None:
                return await self._handoff_result(
                    conversation, intent, trace_id, "approved_template_unavailable"
                )
            if template.locale != self._locale:
                return await self._handoff_result(
                    conversation,
                    intent,
                    trace_id,
                    "approved_template_locale_mismatch",
                )
            command = OutboundCommand(
                dedup_key=f"reply:{event.event_key}",
                conversation_id=conversation.conversation_id,
                recipient_id=event.recipient_id,
                mode=OutboundMode.TEMPLATE,
                text=None,
                template_name=template.name,
                template_locale=template.locale,
                template_variables={"reply": reply},
                policy_decision=decision.reason,
                trace_id=trace_id,
                **_rag_command_fields(rag_result),
            )
        else:
            command = OutboundCommand(
                dedup_key=f"reply:{event.event_key}",
                conversation_id=conversation.conversation_id,
                recipient_id=event.recipient_id,
                mode=OutboundMode.FREEFORM,
                text=reply,
                template_name=None,
                template_locale=None,
                template_variables={},
                policy_decision=decision.reason,
                trace_id=trace_id,
                **_rag_command_fields(rag_result),
            )

        try:
            await self._outbox.enqueue(command)
        except Exception:
            return await self._handoff_result(
                conversation, intent, trace_id, "outbox_unavailable"
            )
        return self._result(
            status="replied",
            intent=intent,
            trace_id=trace_id,
            outbox_created=True,
            rag_result=rag_result,
        )

    async def _enqueue_reply_durable(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        intent: str,
        decision: PolicyDecision,
        reply: str,
        trace_id: str,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        mode = window_mode(conversation, self._now())
        if mode.value == "blocked":
            return await self._commit_durable(
                event=event,
                expected_version=expected_version,
                conversation=conversation,
                command=None,
                status="blocked",
                intent=intent,
                trace_id=trace_id,
                reason="human_active",
                rag_result=rag_result,
            )

        if mode.value == "template":
            if self._templates is None:
                return await self._handoff_result_durable(
                    event,
                    expected_version,
                    conversation,
                    intent,
                    trace_id,
                    "template_required",
                )
            try:
                template = await self._templates.resolve(
                    self._template_name, self._locale
                )
            except Exception:
                template = None
            if template is None:
                return await self._handoff_result_durable(
                    event,
                    expected_version,
                    conversation,
                    intent,
                    trace_id,
                    "approved_template_unavailable",
                )
            if template.locale != self._locale:
                return await self._handoff_result_durable(
                    event,
                    expected_version,
                    conversation,
                    intent,
                    trace_id,
                    "approved_template_locale_mismatch",
                )
            command = OutboundCommand(
                dedup_key=f"reply:{event.event_key}",
                conversation_id=conversation.conversation_id,
                recipient_id=event.recipient_id,
                mode=OutboundMode.TEMPLATE,
                text=None,
                template_name=template.name,
                template_locale=template.locale,
                template_variables={"reply": reply},
                policy_decision=decision.reason,
                trace_id=trace_id,
                **_rag_command_fields(rag_result),
            )
        else:
            command = OutboundCommand(
                dedup_key=f"reply:{event.event_key}",
                conversation_id=conversation.conversation_id,
                recipient_id=event.recipient_id,
                mode=OutboundMode.FREEFORM,
                text=reply,
                template_name=None,
                template_locale=None,
                template_variables={},
                policy_decision=decision.reason,
                trace_id=trace_id,
                **_rag_command_fields(rag_result),
            )

        return await self._commit_durable(
            event=event,
            expected_version=expected_version,
            conversation=conversation,
            command=command,
            status="replied",
            intent=intent,
            trace_id=trace_id,
            outbox_created=True,
            rag_result=rag_result,
        )

    async def _handoff_result(
        self,
        conversation: Conversation,
        intent: str | None,
        trace_id: str,
        reason: str,
        *,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        updated = replace(
            conversation,
            automation_state=AutomationState.HUMAN_ACTIVE,
            version=conversation.version + 1,
        )
        await self._conversations.save(updated)
        accepted = False
        if self._handoff is not None:
            try:
                result = await self._handoff.create(
                    HandoffRequest(
                        request_id=f"handoff:{trace_id}",
                        conversation_id=conversation.conversation_id,
                        reason=reason,
                        requested_by="conversation-worker",
                        requested_at=self._now(),
                        note=f"trace_id={trace_id}",
                    )
                )
                accepted = result.accepted
            except Exception:
                accepted = False
        return self._result(
            status="handed_off" if accepted else "handoff_failed",
            intent=intent,
            trace_id=trace_id,
            handoff_created=accepted,
            reason=reason,
            rag_result=rag_result,
        )

    async def _handoff_result_durable(
        self,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        intent: str | None,
        trace_id: str,
        reason: str,
        *,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        updated = replace(
            conversation,
            automation_state=AutomationState.HUMAN_ACTIVE,
            version=conversation.version + 1,
        )
        request = HandoffRequest(
            request_id=f"handoff:{trace_id}",
            conversation_id=conversation.conversation_id,
            reason=reason,
            requested_by="conversation-worker",
            requested_at=self._now(),
            note=f"trace_id={trace_id}",
        )
        assert self._transaction is not None
        commit = await self._transaction.commit_event(
            event=event,
            expected_version=expected_version,
            conversation=updated,
            command=None,
            handoff_request=request,
        )
        if commit.status == "stale":
            return self._result(
                status="stale_or_duplicate",
                intent=intent,
                trace_id=trace_id,
                reason="conversation_version_conflict",
                retryable=True,
            )
        pending_request = commit.handoff_request
        if commit.status == "duplicate" and pending_request is None:
            return self._result(
                status="stale_or_duplicate",
                intent=intent,
                trace_id=trace_id,
                reason=commit.reason,
            )
        result = await self._deliver_handoff(
            pending_request or request,
            intent=intent,
            trace_id=trace_id,
        )
        return replace(result, **_rag_result_fields(rag_result))

    async def _deliver_handoff(
        self,
        request: HandoffRequest,
        *,
        intent: str | None,
        trace_id: str,
    ) -> WorkerResult:
        accepted = False
        if self._handoff is not None:
            try:
                result = await self._handoff.create(request)
                accepted = result.accepted
            except Exception:
                accepted = False
        if accepted:
            assert self._transaction is not None
            await self._transaction.mark_handoff_complete(request.request_id)
        return self._result(
            status="handed_off" if accepted else "handoff_failed",
            intent=intent,
            trace_id=trace_id,
            handoff_created=accepted,
            reason=request.reason,
            retryable=not accepted,
        )

    async def _commit_durable(
        self,
        *,
        event: InboundEvent,
        expected_version: int,
        conversation: Conversation,
        command: OutboundCommand | None,
        status: str,
        intent: str | None,
        trace_id: str,
        outbox_created: bool = False,
        handoff_created: bool = False,
        reason: str | None = None,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        if conversation.automation_state == AutomationState.HUMAN_ACTIVE:
            command = None
            outbox_created = False
        assert self._transaction is not None
        commit = await self._transaction.commit_event(
            event=event,
            expected_version=expected_version,
            conversation=conversation,
            command=command,
        )
        if commit.status == "stale":
            return self._result(
                status="stale_or_duplicate",
                intent=intent,
                trace_id=trace_id,
                reason="conversation_version_conflict",
                retryable=True,
            )
        if commit.handoff_request is not None:
            return await self._deliver_handoff(
                commit.handoff_request,
                intent=intent,
                trace_id=trace_id,
            )
        if commit.status == "duplicate":
            return self._result(
                status="stale_or_duplicate",
                intent=intent,
                trace_id=trace_id,
                reason=commit.reason,
            )
        return self._result(
            status=status,
            intent=intent,
            trace_id=trace_id,
            outbox_created=outbox_created,
            handoff_created=handoff_created,
            reason=reason,
            rag_result=rag_result,
        )

    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _result(
        *,
        status: str,
        intent: str | None,
        trace_id: str,
        outbox_created: bool = False,
        handoff_created: bool = False,
        reason: str | None = None,
        retryable: bool = False,
        rag_result: RagAnswerResult | None = None,
    ) -> WorkerResult:
        return WorkerResult(
            status=status,
            intent=intent,
            outbox_created=outbox_created,
            handoff_created=handoff_created,
            trace_id=trace_id,
            reason=reason,
            retryable=retryable,
            **_rag_result_fields(rag_result),
        )


def _rag_result_fields(result: RagAnswerResult | None) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "retrieved_chunk_ids": result.retrieved_chunk_ids,
        "supported_chunk_ids": result.supported_chunk_ids,
        "prompt_version": result.prompt_version,
        "model_name": result.model_name,
    }


def _rag_command_fields(result: RagAnswerResult | None) -> dict[str, object]:
    return _rag_result_fields(result)


class OutboxDispatcher:
    def __init__(
        self,
        outbox: Any,
        whatsapp: WhatsAppPort,
        *,
        clock: Callable[[], datetime] | None = None,
        lease_seconds: int = 60,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        self._outbox = outbox
        self._whatsapp = whatsapp
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts

    async def dispatch_once(self) -> DispatchResult:
        record = await self._outbox.claim_pending(
            self._clock(),
            self._lease_seconds,
            max_attempts=self._max_attempts,
        )
        if record is None:
            return DispatchResult(status="idle")

        try:
            send_result = await self._whatsapp.send(record.command)
        except Exception as exc:
            retryable = _is_retryable_exception(exc)
            return await self._finish_failure(record, retryable, _failure_code(exc))

        if not send_result.accepted:
            retryable = _is_retryable_code(send_result.error_code)
            return await self._finish_failure(
                record,
                retryable,
                send_result.error_code or "provider_rejected",
            )

        await self._outbox.mark_sent(
            record.command.dedup_key,
            send_result.provider_message_id,
            attempts=record.attempts,
            claimed_at=record.claimed_at,
        )
        return DispatchResult(
            status="sent",
            dedup_key=record.command.dedup_key,
            provider_message_id=send_result.provider_message_id,
            attempts=record.attempts,
        )

    async def _finish_failure(
        self,
        record: Any,
        retryable: bool,
        error: str,
    ) -> DispatchResult:
        if retryable and record.attempts < self._max_attempts:
            await self._outbox.mark_retryable(
                record.command.dedup_key,
                error,
                attempts=record.attempts,
                claimed_at=record.claimed_at,
            )
            return DispatchResult(
                status="retryable",
                dedup_key=record.command.dedup_key,
                attempts=record.attempts,
                error=error,
            )
        await self._outbox.mark_dead_letter(
            record.command.dedup_key,
            error,
            attempts=record.attempts,
            claimed_at=record.claimed_at,
        )
        return DispatchResult(
            status="dead_letter",
            dedup_key=record.command.dedup_key,
            attempts=record.attempts,
            error=error,
        )


class OutboxReclaimer:
    def __init__(
        self,
        outbox: OutboxPort,
        clock: Callable[[], datetime],
        *,
        max_attempts: int | None = None,
    ) -> None:
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._outbox = outbox
        self._clock = clock
        self._max_attempts = max_attempts

    async def reclaim_once(self) -> int:
        if self._max_attempts is None:
            return await self._outbox.reclaim_expired(self._clock())
        return await self._outbox.reclaim_expired(
            self._clock(), max_attempts=self._max_attempts
        )


def _trace_id(event_key: str) -> str:
    return f"trace-{hashlib.sha256(event_key.encode('utf-8')).hexdigest()[:20]}"


def _is_retryable_exception(error: Exception) -> bool:
    return isinstance(
        error, (RetryableProviderError, TimeoutError, ConnectionError)
    ) or bool(getattr(error, "retryable", False))


def _is_retryable_code(error_code: str | None) -> bool:
    if error_code is None:
        return False
    normalized = error_code.casefold()
    return normalized in {
        "timeout",
        "timed_out",
        "rate_limit",
        "rate_limited",
        "temporarily_unavailable",
        "service_unavailable",
        "network_error",
        "429",
        "500",
        "502",
        "503",
        "504",
    }


def _failure_code(error: Exception) -> str:
    code = getattr(error, "error_code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ConnectionError):
        return "network_error"
    if isinstance(error, PermanentProviderError):
        return "permanent_provider_error"
    return type(error).__name__
