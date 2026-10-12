from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from whatsapp_support.domain import (
    AutomationState,
    Conversation,
    EventType,
    Evidence,
    IdentityState,
    IdentityResult,
    IdentityProof,
    InboundEvent,
    ModelRequest,
    OutboundCommand,
    OutboundMode,
    OutboxRecord,
    ProviderSendResult,
    RagAnswerResult,
    StructuredModelResult,
    TemplateDefinition,
    ToolResult,
)
from whatsapp_support.fakes import (
    FakeBusinessTools,
    FakeHandoff,
    FakeIdentity,
    FakeKnowledge,
    FakeModel,
    FakeTemplateCatalog,
    FakeWhatsApp,
    InMemoryConversationStore,
    InMemoryOutboxStore,
)
from whatsapp_support.policy import IntentRouter, SafetyPolicy
from whatsapp_support.ecommerce_tools import EcommerceToolRegistry, FakeEcommerceTools
from whatsapp_support.orchestration import ConversationOrchestrator
from whatsapp_support.replies import ReplyComposer
from whatsapp_support.worker import ConversationWorker, OutboxDispatcher, OutboxReclaimer


NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)


class StaticRagAnswer:
    def __init__(
        self,
        *,
        status: str = "reply",
        reply: str = "The backpack shell is recycled nylon.",
        reason: str = "grounded_answer",
    ) -> None:
        self.status = status
        self.reply = reply
        self.reason = reason

    async def answer(
        self, event: InboundEvent, locale: str, trace_id: str
    ) -> RagAnswerResult:
        return RagAnswerResult(
            status=self.status,
            reply=self.reply if self.status == "reply" else None,
            locale=locale,
            trace_id=trace_id,
            retrieved_chunk_ids=("chunk-1",),
            supported_chunk_ids=("chunk-1",),
            prompt_version="prompt-v1",
            model_name="answer-v1",
            reason=self.reason,
        )


def make_event(
    *,
    event_key: str = "event-1",
    text: str = "Where is my order?",
    occurred_at: datetime = NOW,
    identity_proof: IdentityProof | None = None,
    recipient_id: str = "recipient-1",
) -> InboundEvent:
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id=recipient_id,
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=occurred_at,
        received_at=occurred_at,
        payload_digest=f"digest-{event_key}",
        text=text,
        identity_proof=identity_proof,
    )


def make_command(
    dedup_key: str = "command-1", *, recipient_id: str = "recipient-1"
) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conversation-1",
        recipient_id=recipient_id,
        mode=OutboundMode.FREEFORM,
        text="hello",
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=f"trace-{dedup_key}",
    )


def make_worker(
    *,
    clock=NOW,
    rag_answer=None,
) -> tuple[
    ConversationWorker,
    InMemoryConversationStore,
    InMemoryOutboxStore,
    FakeBusinessTools,
    FakeIdentity,
    FakeKnowledge,
    FakeModel,
    FakeTemplateCatalog,
    FakeHandoff,
]:
    conversations = InMemoryConversationStore()
    outbox = InMemoryOutboxStore()
    tools = FakeBusinessTools()
    identity = FakeIdentity()
    knowledge = FakeKnowledge()
    model = FakeModel()
    templates = FakeTemplateCatalog()
    handoff = FakeHandoff()
    worker = ConversationWorker(
        conversations=conversations,
        outbox=outbox,
        tools=tools,
        identity=identity,
        knowledge=knowledge,
        model=model,
        templates=templates,
        handoff=handoff,
        clock=lambda: clock,
        rag_answer=rag_answer,
    )
    return (
        worker,
        conversations,
        outbox,
        tools,
        identity,
        knowledge,
        model,
        templates,
        handoff,
    )


def test_worker_product_information_creates_outbox_reply() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        outbox = InMemoryOutboxStore()
        ecommerce = FakeEcommerceTools(clock=lambda: NOW)
        orchestrator = ConversationOrchestrator(
            tools=ecommerce,
            identity=None,
            router=IntentRouter(),
            policy=SafetyPolicy(),
            composer=ReplyComposer(),
            registry=EcommerceToolRegistry(
                {
                    "get_product_info": ecommerce,
                    "get_product_inventory": ecommerce,
                }
            ),
            clock=lambda: NOW,
            locale="en",
        )
        worker = ConversationWorker(
            conversations,
            outbox,
            FakeBusinessTools(),
            FakeIdentity(),
            FakeKnowledge(),
            FakeModel(),
            FakeTemplateCatalog(),
            FakeHandoff(),
            clock=lambda: NOW,
            orchestrator=orchestrator,
            rag_answer=StaticRagAnswer(),
        )

        result = await worker.process(make_event(text="What is product sku-1?"))

        assert result.status == "replied"
        assert (await outbox.records())[0].command.text == (
            "The backpack shell is recycled nylon."
        )

    asyncio.run(scenario())


def test_worker_without_orchestrator_composes_ecommerce_output_safely() -> None:
    async def scenario() -> None:
        worker, _, outbox, tools, _, _, _, _, _ = make_worker()
        tools.results_queue.append(
            ToolResult(
                request_id="unused",
                tool_name="get_product_inventory",
                success=True,
                output_text="INTERNAL RAW TOOL OUTPUT",
                output_data={
                    "product_id": "sku-1",
                    "name": "Everyday Backpack",
                    "available": True,
                    "quantity": 7,
                },
                completed_at=NOW,
            )
        )

        result = await worker.process(make_event(text="inventory sku-1"))

        assert isinstance(worker._orchestrator, ConversationOrchestrator)
        assert result.status == "replied"
        command = (await outbox.records())[0].command
        assert command.text == (
            "Everyday Backpack is in stock (7 available)."
        )
        assert "INTERNAL RAW TOOL OUTPUT" not in command.text

    asyncio.run(scenario())


def test_worker_uses_orchestrator_for_verified_order_query() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        outbox = InMemoryOutboxStore()
        identity = FakeIdentity()
        identity.results_queue.append(
            IdentityResult(
                verified=True,
                scopes=frozenset({"order:read"}),
                subject_id="customer-1",
                status="verified",
                expires_at=NOW + timedelta(minutes=5),
            )
        )
        ecommerce = FakeEcommerceTools(clock=lambda: NOW)
        orchestrator = ConversationOrchestrator(
            tools=ecommerce,
            identity=identity,
            router=IntentRouter(),
            policy=SafetyPolicy(),
            composer=ReplyComposer(),
            registry=EcommerceToolRegistry(
                {
                    "get_order_status": ecommerce,
                    "get_shipping_tracking": ecommerce,
                    "check_return_eligibility": ecommerce,
                }
            ),
            clock=lambda: NOW,
            locale="en",
        )
        legacy_tools = FakeBusinessTools()
        worker = ConversationWorker(
            conversations,
            outbox,
            legacy_tools,
            identity,
            FakeKnowledge(),
            FakeModel(),
            FakeTemplateCatalog(),
            FakeHandoff(),
            clock=lambda: NOW,
            orchestrator=orchestrator,
            rag_answer=StaticRagAnswer(),
        )

        result = await worker.process(
            make_event(
                text="Where is my order order-1?",
                identity_proof=IdentityProof(kind="otp", value="123456"),
            )
        )

        record = (await outbox.records())[0]
        assert result.outbox_created is True
        assert record.command.text is not None
        assert "Order order-1 is shipped" in record.command.text
        assert legacy_tools.calls == []
        assert len(identity.calls) == 1

    asyncio.run(scenario())


def test_same_product_event_keeps_one_reply_dedup_key_after_reprocessing() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        outbox = InMemoryOutboxStore()
        ecommerce = FakeEcommerceTools(clock=lambda: NOW)
        orchestrator = ConversationOrchestrator(
            tools=ecommerce,
            identity=None,
            router=IntentRouter(),
            policy=SafetyPolicy(),
            composer=ReplyComposer(),
            registry=EcommerceToolRegistry(
                {
                    "get_product_info": ecommerce,
                    "get_product_inventory": ecommerce,
                }
            ),
            clock=lambda: NOW,
            locale="en",
        )
        worker = ConversationWorker(
            conversations,
            outbox,
            FakeBusinessTools(),
            FakeIdentity(),
            FakeKnowledge(),
            FakeModel(),
            FakeTemplateCatalog(),
            FakeHandoff(),
            clock=lambda: NOW,
            orchestrator=orchestrator,
            rag_answer=StaticRagAnswer(),
        )
        event = make_event(event_key="same", text="product sku-1")

        first = await worker.process(event)
        second = await worker.process(event)

        assert first.outbox_created is True
        assert second.outbox_created is False
        records = await outbox.records()
        assert [record.command.dedup_key for record in records] == ["reply:same"]

    asyncio.run(scenario())


def test_intent_router_covers_read_only_and_human_risk_intents() -> None:
    router = IntentRouter()

    assert router.route("What material is this product made from?") == "product_policy"
    assert router.route("订单状态") == "order_status"
    assert router.route("Is this item in stock?") == "inventory"
    assert router.route("请查物流单号") == "shipping"
    assert router.route("Am I eligible for a return?") == "return_eligibility"
    assert router.route("I want a refund") == "refund"
    assert router.route("please connect me to a human") == "human"
    assert router.route("tell me a joke") is None


def test_router_prioritizes_personal_live_signals_over_static_policy_words() -> None:
    router = IntentRouter()

    assert (
        router.route("Can I return order-1 under your return policy?")
        == "return_eligibility"
    )
    assert (
        router.route("Tracking order-1 under the shipping policy") == "shipping"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("change address for product sku-1", "change_address"),
        ("cancel order for product sku-1", "cancel_order"),
        ("complaint about product sku-1", "complaint"),
        ("human help with product sku-1", "human"),
    ],
)
def test_write_risk_intents_take_precedence_over_product_lookup(
    text: str, expected: str
) -> None:
    assert IntentRouter().route(text) == expected


def test_safety_policy_requires_identity_for_protected_tools_and_human_for_writes() -> None:
    policy = SafetyPolicy()
    unverified = Conversation(
        conversation_id="conversation-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
    )
    verified = Conversation(
        conversation_id="conversation-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        identity_state=IdentityState.VERIFIED,
        identity_subject_id="customer-1",
        identity_scopes=frozenset({"order:read"}),
        identity_expires_at=NOW + timedelta(minutes=5),
    )

    order_decision = policy.authorize(unverified, "order_status", now=NOW)
    assert order_decision.allowed is False
    assert order_decision.handoff is True
    assert order_decision.required_scope == "order:read"
    assert policy.authorize(verified, "order_status", now=NOW).allowed is True
    assert policy.authorize(unverified, "refund", now=NOW).handoff is True
    assert policy.authorize(unverified, "inventory", now=NOW).allowed is True


@pytest.mark.parametrize(
    "grant",
    [
        {
            "identity_subject_id": "customer-1",
            "identity_scopes": frozenset({"shipping:read"}),
            "identity_expires_at": NOW + timedelta(minutes=5),
        },
        {
            "identity_subject_id": "customer-1",
            "identity_scopes": frozenset({"order:read"}),
            "identity_expires_at": NOW,
        },
        {
            "identity_subject_id": None,
            "identity_scopes": frozenset({"order:read"}),
            "identity_expires_at": NOW + timedelta(minutes=5),
        },
    ],
)
def test_safety_policy_rejects_incomplete_expired_or_wrong_scope_grants(
    grant: dict[str, object],
) -> None:
    conversation = Conversation(
        conversation_id="conversation-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
        identity_state=IdentityState.VERIFIED,
        **grant,
    )

    decision = SafetyPolicy().authorize(conversation, "order_status", now=NOW)

    assert decision.allowed is False
    assert decision.handoff is True
    assert decision.reason == "identity_required"


def test_worker_does_not_call_protected_tool_before_identity_verification() -> None:
    async def scenario() -> None:
        worker, _, outbox, tools, _, _, _, _, handoff = make_worker()

        result = await worker.process(make_event(text="Where is my order?"))

        assert result.handoff_created is True
        assert result.outbox_created is False
        assert tools.calls == []
        assert await outbox.records() == []
        assert len(handoff.calls) == 1

    asyncio.run(scenario())


def test_worker_verifies_identity_then_calls_allowlisted_tool_and_enqueues_once() -> None:
    async def scenario() -> None:
        worker, conversations, outbox, tools, identity, _, _, _, handoff = make_worker()
        identity.results_queue.append(
            IdentityResult(
                verified=True,
                scopes=frozenset({"order:read"}),
                subject_id="customer-1",
                status="verified",
                expires_at=NOW + timedelta(minutes=5),
            )
        )
        tools.results_queue.append(
            ToolResult(
                request_id="unused",
                tool_name="get_order_status",
                success=True,
                output_text="Your order is on the way.",
                output_data={
                    "order_id": "order-1",
                    "status": "shipped",
                    "updated_at": "2026-08-27T10:00:00Z",
                },
                completed_at=NOW,
            )
        )
        event = make_event(
            text="Where is my order?",
            identity_proof=IdentityProof(kind="otp", value="123456"),
        )

        first = await worker.process(event)
        second = await worker.process(event)

        assert first.outbox_created is True
        assert second.outbox_created is False
        assert second.status == "stale_or_duplicate"
        assert len(identity.calls) == 1
        assert len(tools.calls) == 1
        assert tools.calls[0].tool_name == "get_order_status"
        assert tools.calls[0].arguments == {"subject_id": "customer-1"}
        assert len(await outbox.records()) == 1
        assert handoff.calls == []
        persisted = await conversations.get("conversation-1")
        assert persisted is not None
        assert persisted.identity_subject_id == "customer-1"
        assert persisted.identity_scopes == frozenset({"order:read"})
        assert persisted.identity_expires_at == NOW + timedelta(minutes=5)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("scopes", "expires_at"),
    [
        (frozenset({"shipping:read"}), NOW + timedelta(minutes=5)),
        (frozenset({"order:read"}), NOW - timedelta(seconds=1)),
    ],
)
def test_worker_does_not_shortcut_verification_for_wrong_scope_or_expired_grant(
    scopes: frozenset[str],
    expires_at: datetime,
) -> None:
    async def scenario() -> None:
        worker, conversations, outbox, tools, identity, _, _, _, handoff = make_worker()
        await conversations.save(
            Conversation(
                conversation_id="conversation-1",
                contact_id="contact-1",
                phone_number_id="phone-1",
                identity_state=IdentityState.VERIFIED,
                identity_subject_id="customer-1",
                identity_scopes=scopes,
                identity_expires_at=expires_at,
            )
        )

        result = await worker.process(make_event(text="Where is my order?"))

        assert result.handoff_created is True
        assert identity.calls == []
        assert tools.calls == []
        assert await outbox.records() == []
        assert len(handoff.calls) == 1

    asyncio.run(scenario())


def test_human_active_conversation_never_creates_robot_outbox() -> None:
    async def scenario() -> None:
        worker, conversations, outbox, tools, _, _, _, _, handoff = make_worker()
        await conversations.save(
            Conversation(
                conversation_id="conversation-1",
                contact_id="contact-1",
                phone_number_id="phone-1",
                automation_state=AutomationState.HUMAN_ACTIVE,
            )
        )

        result = await worker.process(make_event(text="What is the price?"))

        assert result.status == "human_active"
        assert result.outbox_created is False
        assert tools.calls == []
        assert await outbox.records() == []
        assert handoff.calls == []

    asyncio.run(scenario())


def test_product_policy_answer_requires_evidence_and_valid_model_reply() -> None:
    async def scenario() -> None:
        worker, _, outbox, _, _, knowledge, model, _, handoff = make_worker(
            rag_answer=StaticRagAnswer(reply="Returns are accepted within 30 days.")
        )
        knowledge.results_queue.append(
            [
                Evidence(
                    evidence_id="policy-1",
                    kind="policy",
                    source="fixture",
                    collected_at=NOW,
                    detail="Returns are accepted within 30 days.",
                )
            ]
        )
        model.results_queue.append(
            StructuredModelResult(
                trace_id="model-trace",
                model_name="fixture-model",
                prompt_version="v1",
                content={"reply": "Returns are accepted within 30 days."},
                raw_text='{"reply":"Returns are accepted within 30 days."}',
                created_at=NOW,
            )
        )

        event = make_event(
            text="What is your return policy?", recipient_id="wa-worker-42"
        )
        result = await worker.process(event)

        records = await outbox.records()
        assert result.outbox_created is True
        assert knowledge.calls == []
        assert model.calls == []
        assert records[0].command.text == "Returns are accepted within 30 days."
        assert records[0].command.mode is OutboundMode.FREEFORM
        assert records[0].command.recipient_id == event.recipient_id
        assert handoff.calls == []

    asyncio.run(scenario())


def test_worker_hands_off_when_evidence_is_missing_or_model_schema_is_invalid() -> None:
    async def scenario() -> None:
        missing_worker, _, missing_outbox, _, _, _, _, _, missing_handoff = make_worker(
            rag_answer=StaticRagAnswer(
                status="handoff", reason="insufficient_evidence"
            )
        )
        missing = await missing_worker.process(make_event(text="What is your return policy?"))
        assert missing.handoff_created is True
        assert await missing_outbox.records() == []
        assert len(missing_handoff.calls) == 1

        invalid_worker, _, invalid_outbox, _, _, invalid_knowledge, invalid_model, _, invalid_handoff = make_worker(
            rag_answer=StaticRagAnswer(
                status="handoff", reason="model_schema_invalid"
            )
        )
        invalid_knowledge.results_queue.append(
            [
                Evidence(
                    evidence_id="policy-1",
                    kind="policy",
                    source="fixture",
                    collected_at=NOW,
                    detail="some evidence",
                )
            ]
        )
        invalid_model.results_queue.append(
            StructuredModelResult(
                trace_id="model-trace",
                model_name="fixture-model",
                prompt_version="v1",
                content={"unexpected": "no reply"},
                raw_text="{}",
                created_at=NOW,
            )
        )
        invalid = await invalid_worker.process(make_event(text="What is your return policy?"))
        assert invalid.handoff_created is True
        assert await invalid_outbox.records() == []
        assert len(invalid_handoff.calls) == 1

    asyncio.run(scenario())


def test_expired_window_uses_audited_template_mode() -> None:
    async def scenario() -> None:
        worker, _, outbox, _, _, knowledge, model, templates, _ = make_worker(
            clock=NOW + timedelta(hours=25),
            rag_answer=StaticRagAnswer(reply="Policy answer"),
        )
        knowledge.results_queue.append(
            [
                Evidence(
                    evidence_id="policy-1",
                    kind="policy",
                    source="fixture",
                    collected_at=NOW,
                    detail="Policy evidence",
                )
            ]
        )
        model.results_queue.append(
            StructuredModelResult(
                trace_id="model-trace",
                model_name="fixture-model",
                prompt_version="v1",
                content={"reply": "Policy answer"},
                raw_text='{"reply":"Policy answer"}',
                created_at=NOW,
            )
        )
        templates.results_queue.append(
            TemplateDefinition(
                name="customer_service_reply",
                locale="en",
                category="utility",
                body_template="{{reply}}",
                variable_names=("reply",),
            )
        )

        result = await worker.process(make_event(text="What is your return policy?"))

        record = (await outbox.records())[0]
        assert result.outbox_created is True
        assert record.command.mode is OutboundMode.TEMPLATE
        assert record.command.text is None
        assert record.command.template_name == "customer_service_reply"
        assert record.command.template_variables == {"reply": "Policy answer"}

    asyncio.run(scenario())


def test_expired_window_hands_off_when_template_locale_does_not_match() -> None:
    async def scenario() -> None:
        worker, _, outbox, _, _, knowledge, model, templates, handoff = make_worker(
            clock=NOW + timedelta(hours=25),
            rag_answer=StaticRagAnswer(reply="Policy answer"),
        )
        knowledge.results_queue.append(
            [
                Evidence(
                    evidence_id="policy-1",
                    kind="policy",
                    source="fixture",
                    collected_at=NOW,
                    detail="Policy evidence",
                )
            ]
        )
        model.results_queue.append(
            StructuredModelResult(
                trace_id="model-trace",
                model_name="fixture-model",
                prompt_version="v1",
                content={"reply": "Policy answer"},
                raw_text='{"reply":"Policy answer"}',
                created_at=NOW,
            )
        )
        templates.results_queue.append(
            TemplateDefinition(
                name="customer_service_reply",
                locale="zh-CN",
                category="utility",
                body_template="{{reply}}",
                variable_names=("reply",),
            )
        )

        result = await worker.process(make_event(text="What is your return policy?"))

        assert result.status == "handed_off"
        assert result.reason == "approved_template_locale_mismatch"
        assert await outbox.records() == []
        assert len(handoff.calls) == 1

    asyncio.run(scenario())


def test_dispatcher_records_provider_id_and_does_not_duplicate_commands() -> None:
    async def scenario() -> None:
        outbox = InMemoryOutboxStore()
        whatsapp = FakeWhatsApp()
        await outbox.enqueue(make_command("same-command"))
        await outbox.enqueue(make_command("same-command"))
        whatsapp.results_queue.append(
            ProviderSendResult(
                provider_message_id="wamid-sent",
                accepted=True,
                sent_at=NOW,
                status="accepted",
            )
        )
        dispatcher = OutboxDispatcher(outbox, whatsapp)

        result = await dispatcher.dispatch_once()
        records = await outbox.records()

        assert result.status == "sent"
        assert result.provider_message_id == "wamid-sent"
        assert len(whatsapp.calls) == 1
        assert len(records) == 1
        assert records[0].state == "SENT"
        assert records[0].provider_message_id == "wamid-sent"

    asyncio.run(scenario())


def test_dispatcher_claims_with_configured_clock_and_lease() -> None:
    class RecordingOutbox:
        def __init__(self) -> None:
            self.claim_args: list[
                tuple[datetime | None, int | None, int | None]
            ] = []
            self.sent: list[tuple[str, str, int | None, datetime | None]] = []

        async def claim_pending(
            self,
            now: datetime | None = None,
            lease_seconds: int | None = None,
            max_attempts: int | None = None,
        ) -> OutboxRecord | None:
            self.claim_args.append((now, lease_seconds, max_attempts))
            return OutboxRecord(
                command=make_command("lease-dispatch"),
                attempts=2,
                claimed_at=NOW,
            )

        async def mark_sent(
            self,
            dedup_key: str,
            provider_message_id: str,
            *,
            attempts: int | None = None,
            claimed_at: datetime | None = None,
        ) -> None:
            self.sent.append((dedup_key, provider_message_id, attempts, claimed_at))

    async def scenario() -> None:
        outbox = RecordingOutbox()
        whatsapp = FakeWhatsApp()
        whatsapp.results_queue.append(
            ProviderSendResult(
                provider_message_id="wamid-lease",
                accepted=True,
                sent_at=NOW,
                status="accepted",
            )
        )
        dispatcher = OutboxDispatcher(
            outbox,
            whatsapp,
            clock=lambda: NOW,
            lease_seconds=45,
        )

        result = await dispatcher.dispatch_once()

        assert result.status == "sent"
        assert outbox.claim_args == [(NOW, 45, 3)]
        assert outbox.sent == [("lease-dispatch", "wamid-lease", 2, NOW)]
        assert len(whatsapp.calls) == 1

    asyncio.run(scenario())


def test_dispatcher_retries_transient_failure_then_sends() -> None:
    async def scenario() -> None:
        outbox = InMemoryOutboxStore()
        whatsapp = FakeWhatsApp()
        await outbox.enqueue(make_command("retry-command"))
        whatsapp.errors_queue.append(TimeoutError("provider timeout"))
        whatsapp.results_queue.append(
            ProviderSendResult(
                provider_message_id="wamid-retry",
                accepted=True,
                sent_at=NOW,
                status="accepted",
            )
        )
        dispatcher = OutboxDispatcher(outbox, whatsapp)

        first = await dispatcher.dispatch_once()
        second = await dispatcher.dispatch_once()

        assert first.status == "retryable"
        assert first.dedup_key == "retry-command"
        assert second.status == "sent"
        record = (await outbox.records())[0]
        assert record.command.dedup_key == "retry-command"
        assert record.attempts == 2
        assert record.state == "SENT"

    asyncio.run(scenario())


def test_dispatcher_passes_claim_fence_to_retryable_and_dead_letter_updates() -> None:
    class RecordingOutbox:
        def __init__(self, record: OutboxRecord) -> None:
            self.record = record
            self.retryable: list[tuple[str, str, int | None, datetime | None]] = []
            self.dead_letters: list[tuple[str, str, int | None, datetime | None]] = []

        async def claim_pending(
            self,
            now: datetime | None = None,
            lease_seconds: int | None = None,
            max_attempts: int | None = None,
        ) -> OutboxRecord | None:
            del now, lease_seconds, max_attempts
            return self.record

        async def mark_retryable(
            self,
            dedup_key: str,
            reason: str,
            *,
            attempts: int | None = None,
            claimed_at: datetime | None = None,
        ) -> None:
            self.retryable.append((dedup_key, reason, attempts, claimed_at))

        async def mark_dead_letter(
            self,
            dedup_key: str,
            reason: str,
            *,
            attempts: int | None = None,
            claimed_at: datetime | None = None,
        ) -> None:
            self.dead_letters.append((dedup_key, reason, attempts, claimed_at))

    async def scenario() -> None:
        retry_outbox = RecordingOutbox(
            OutboxRecord(
                command=make_command("retry-fenced"),
                attempts=2,
                claimed_at=NOW,
            )
        )
        retry_whatsapp = FakeWhatsApp()
        retry_whatsapp.errors_queue.append(TimeoutError("timeout"))
        retry_dispatcher = OutboxDispatcher(
            retry_outbox,
            retry_whatsapp,
            clock=lambda: NOW,
            max_attempts=3,
        )

        retry_result = await retry_dispatcher.dispatch_once()

        dead_outbox = RecordingOutbox(
            OutboxRecord(
                command=make_command("dead-fenced"),
                attempts=2,
                claimed_at=NOW,
            )
        )
        dead_whatsapp = FakeWhatsApp()
        dead_whatsapp.errors_queue.append(TimeoutError("timeout"))
        dead_dispatcher = OutboxDispatcher(
            dead_outbox,
            dead_whatsapp,
            clock=lambda: NOW,
            max_attempts=2,
        )

        dead_result = await dead_dispatcher.dispatch_once()

        assert retry_result.status == "retryable"
        assert retry_outbox.retryable == [("retry-fenced", "timeout", 2, NOW)]
        assert retry_outbox.dead_letters == []
        assert dead_result.status == "dead_letter"
        assert dead_outbox.dead_letters == [("dead-fenced", "timeout", 2, NOW)]
        assert dead_outbox.retryable == []

    asyncio.run(scenario())


def test_outbox_reclaimer_reclaims_without_sending_messages() -> None:
    class RecordingOutbox:
        def __init__(self) -> None:
            self.calls: list[tuple[str, datetime | None]] = []

        async def reclaim_expired(self, now: datetime) -> int:
            self.calls.append(("reclaim_expired", now))
            return 7

        async def send(self) -> None:
            self.calls.append(("send", None))
            raise AssertionError("OutboxReclaimer must not send")

    async def scenario() -> None:
        outbox = RecordingOutbox()
        reclaimer = OutboxReclaimer(outbox, clock=lambda: NOW + timedelta(seconds=60))

        reclaimed = await reclaimer.reclaim_once()

        assert reclaimed == 7
        assert outbox.calls == [("reclaim_expired", NOW + timedelta(seconds=60))]

    asyncio.run(scenario())


def test_dispatcher_dead_letters_permanent_failure_and_bounded_retries() -> None:
    async def scenario() -> None:
        permanent_outbox = InMemoryOutboxStore()
        permanent_whatsapp = FakeWhatsApp()
        await permanent_outbox.enqueue(make_command("permanent-command"))
        permanent_whatsapp.errors_queue.append(ValueError("invalid template"))
        permanent_dispatcher = OutboxDispatcher(permanent_outbox, permanent_whatsapp)

        permanent_result = await permanent_dispatcher.dispatch_once()
        permanent_record = (await permanent_outbox.records())[0]
        assert permanent_result.status == "dead_letter"
        assert permanent_record.state == "DEAD_LETTER"

        bounded_outbox = InMemoryOutboxStore()
        bounded_whatsapp = FakeWhatsApp()
        await bounded_outbox.enqueue(make_command("bounded-command"))
        bounded_whatsapp.errors_queue.extend(
            [TimeoutError("timeout-1"), TimeoutError("timeout-2")]
        )
        bounded_dispatcher = OutboxDispatcher(
            bounded_outbox, bounded_whatsapp, max_attempts=2
        )

        first = await bounded_dispatcher.dispatch_once()
        second = await bounded_dispatcher.dispatch_once()
        bounded_record = (await bounded_outbox.records())[0]
        assert first.status == "retryable"
        assert second.status == "dead_letter"
        assert bounded_record.attempts == 2
        assert bounded_record.state == "DEAD_LETTER"

    asyncio.run(scenario())
