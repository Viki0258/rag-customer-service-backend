import asyncio
from datetime import UTC, datetime, timedelta

from whatsapp_support.domain import (
    Conversation,
    EventType,
    IdentityProof,
    IdentityResult,
    IdentityState,
    InboundEvent,
    ToolResult,
)
from whatsapp_support.ecommerce_tools import EcommerceToolRegistry, FakeEcommerceTools
from whatsapp_support.orchestration import ConversationOrchestrator
from whatsapp_support.policy import IntentRouter, SafetyPolicy
from whatsapp_support.replies import ReplyComposer


NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)


def make_status_event() -> InboundEvent:
    return InboundEvent(
        event_key="status-1",
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id="recipient-1",
        provider_message_id="wamid-status-1",
        event_type=EventType.STATUS,
        occurred_at=NOW,
        received_at=NOW,
        payload_digest="digest-status-1",
    )


def make_event(
    text: str,
    *,
    proof: IdentityProof | None = None,
) -> InboundEvent:
    return InboundEvent(
        event_key="message-1",
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id="recipient-1",
        provider_message_id="wamid-message-1",
        event_type=EventType.MESSAGE,
        occurred_at=NOW,
        received_at=NOW,
        payload_digest="digest-message-1",
        text=text,
        identity_proof=proof,
    )


def make_conversation() -> Conversation:
    return Conversation(
        conversation_id="conversation-1",
        contact_id="contact-1",
        phone_number_id="phone-1",
    )


class RecordingBusinessTools:
    def __init__(self) -> None:
        self.calls = []
        self.results: list[ToolResult] = []

    async def execute(self, request):
        self.calls.append(request)
        return self.results.pop(0)


class QueueIdentity:
    def __init__(self, result: IdentityResult) -> None:
        self.result = result
        self.calls = []

    async def verify(self, conversation_id, proof):
        self.calls.append((conversation_id, proof))
        return self.result


def make_orchestrator(*, tools=None, identity=None) -> ConversationOrchestrator:
    registry = None
    if tools is not None:
        registry = EcommerceToolRegistry(
            {
                "get_product_info": tools,
                "get_product_inventory": tools,
                "get_order_status": tools,
                "get_shipping_tracking": tools,
                "check_return_eligibility": tools,
            }
        )
    return ConversationOrchestrator(
        tools=tools,
        identity=identity,
        router=IntentRouter(),
        policy=SafetyPolicy(),
        composer=ReplyComposer(),
        registry=registry,
        clock=lambda: NOW,
        locale="en",
    )


def test_orchestrator_ignores_non_message_event_without_side_effects() -> None:
    async def scenario() -> None:
        orchestrator = ConversationOrchestrator(
            tools=None,
            identity=None,
            router=IntentRouter(),
            policy=SafetyPolicy(),
            composer=ReplyComposer(),
            clock=lambda: NOW,
        )

        result = await orchestrator.run(make_status_event(), make_conversation())

        assert result.status == "ignored"
        assert result.reply is None
        assert result.intent is None

    asyncio.run(scenario())


def test_orchestrator_does_not_call_protected_tool_before_identity() -> None:
    async def scenario() -> None:
        tools = RecordingBusinessTools()
        identity = QueueIdentity(
            IdentityResult(
                verified=False,
                scopes=frozenset(),
                subject_id="",
                status="failed",
            )
        )

        result = await make_orchestrator(tools=tools, identity=identity).run(
            make_event("Where is my order?"), make_conversation()
        )

        assert result.status == "handoff"
        assert result.handoff_reason in {"identity_required", "identity_not_authorized"}
        assert tools.calls == []

    asyncio.run(scenario())


def test_orchestrator_calls_verified_order_tool_and_returns_reply() -> None:
    async def scenario() -> None:
        tools = RecordingBusinessTools()
        identity = QueueIdentity(
            IdentityResult(
                verified=True,
                scopes=frozenset({"order:read"}),
                subject_id="customer-1",
                status="verified",
                expires_at=NOW + timedelta(minutes=5),
            )
        )
        tools.results.append(
            ToolResult(
                request_id="unused",
                tool_name="get_order_status",
                success=True,
                output_text="ok",
                output_data={
                    "order_id": "order-1",
                    "status": "shipped",
                    "updated_at": "2026-08-27T10:00:00Z",
                },
                completed_at=NOW,
            )
        )

        result = await make_orchestrator(tools=tools, identity=identity).run(
            make_event(
                "Where is my order order-1?",
                proof=IdentityProof(kind="otp", value="123456"),
            ),
            make_conversation(),
        )

        assert result.status == "reply"
        assert result.intent == "order_status"
        assert result.reply is not None
        assert tools.calls[0].tool_name == "get_order_status"
        assert tools.calls[0].arguments["order_id"] == "order-1"

    asyncio.run(scenario())


def test_orchestrator_uses_verified_subject_when_order_id_is_omitted() -> None:
    async def scenario() -> None:
        identity = QueueIdentity(
            IdentityResult(
                verified=True,
                scopes=frozenset({"order:read"}),
                subject_id="customer-1",
                status="verified",
                expires_at=NOW + timedelta(minutes=5),
            )
        )
        tools = FakeEcommerceTools(clock=lambda: NOW)

        result = await make_orchestrator(tools=tools, identity=identity).run(
            make_event(
                "Where is my order?",
                proof=IdentityProof(kind="otp", value="123456"),
            ),
            make_conversation(),
        )

        assert result.status == "reply"
        assert result.reply is not None
        assert "Order order-1 is shipped" in result.reply.text

    asyncio.run(scenario())


def test_orchestrator_uses_persisted_verified_subject_without_new_proof() -> None:
    async def scenario() -> None:
        tools = FakeEcommerceTools(clock=lambda: NOW)
        conversation = Conversation(
            conversation_id="conversation-1",
            contact_id="contact-1",
            phone_number_id="phone-1",
            identity_state=IdentityState.VERIFIED,
            identity_subject_id="customer-1",
            identity_scopes=frozenset({"order:read"}),
            identity_expires_at=NOW + timedelta(minutes=5),
        )

        result = await make_orchestrator(tools=tools).run(
            make_event("Where is my order?"), conversation
        )

        assert result.status == "reply"
        assert result.reply is not None
        assert "Order order-1 is shipped" in result.reply.text

    asyncio.run(scenario())


def test_orchestrator_hands_off_write_risk_without_calling_tools() -> None:
    async def scenario() -> None:
        tools = RecordingBusinessTools()

        result = await make_orchestrator(tools=tools).run(
            make_event("I want a refund"), make_conversation()
        )

        assert result.status == "handoff"
        assert result.handoff_reason == "write_requires_human"
        assert tools.calls == []

    asyncio.run(scenario())
