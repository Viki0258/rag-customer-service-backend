from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from whatsapp_support.domain import ConversationSummary, ConversationTurn, EventType, InboundEvent, OutboundCommand, OutboundMode
from whatsapp_support.memory import (
    InMemoryConversationMemory,
    RollingSummaryPolicy,
    conservative_structured_summary,
)


NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def make_event(*, event_key: str, text: str, occurred_at: datetime) -> InboundEvent:
    return InboundEvent(
        event_key=event_key,
        phone_number_id="phone-1",
        conversation_id="conversation-1",
        contact_id="contact-1",
        recipient_id="recipient-1",
        provider_message_id=f"wamid-{event_key}",
        event_type=EventType.MESSAGE,
        occurred_at=occurred_at,
        received_at=occurred_at,
        payload_digest=f"digest-{event_key}",
        text=text,
    )


def make_command(*, dedup_key: str, text: str, trace_id: str) -> OutboundCommand:
    return OutboundCommand(
        dedup_key=dedup_key,
        conversation_id="conversation-1",
        recipient_id="recipient-1",
        mode=OutboundMode.FREEFORM,
        text=text,
        template_name=None,
        template_locale=None,
        template_variables={},
        policy_decision="allow",
        trace_id=trace_id,
    )


def make_summary(version: int = 1) -> ConversationSummary:
    return ConversationSummary(
        version=version,
        covered_until=NOW,
        data={
            "customer_goal": "goal",
            "confirmed_entities": [],
            "customer_preferences": [],
            "unresolved_questions": [],
            "decisions": [],
            "handoff_state": "none",
        },
    )


def test_memory_deduplicates_customer_and_assistant_records() -> None:
    memory = InMemoryConversationMemory()
    event = make_event(event_key="evt-1", text="hi", occurred_at=NOW)
    command = make_command(dedup_key="reply-1", text="hello", trace_id="trace-1")

    memory.append_customer(event)
    memory.append_customer(event)
    memory.append_assistant(command)
    memory.append_assistant(command)

    context = memory.context("conversation-1")

    assert [turn.customer_text for turn in context.recent_turns] == ["hi"]
    assert [turn.assistant_text for turn in context.recent_turns] == ["hello"]


def test_memory_returns_latest_six_complete_turns_without_splitting_pairs() -> None:
    memory = InMemoryConversationMemory()
    for index in range(7):
        event = make_event(
            event_key=f"evt-{index}",
            text=f"customer-{index}",
            occurred_at=NOW + timedelta(minutes=index * 2),
        )
        command = make_command(
            dedup_key=f"reply-{index}",
            text=f"assistant-{index}",
            trace_id=f"trace-{index}",
        )
        memory.append_customer(event)
        memory.append_assistant(command)

    context = memory.context("conversation-1")

    assert len(context.recent_turns) == 6
    assert [turn.customer_text for turn in context.recent_turns] == [
        "customer-1",
        "customer-2",
        "customer-3",
        "customer-4",
        "customer-5",
        "customer-6",
    ]
    assert [turn.assistant_text for turn in context.recent_turns] == [
        "assistant-1",
        "assistant-2",
        "assistant-3",
        "assistant-4",
        "assistant-5",
        "assistant-6",
    ]


def test_memory_summarizes_oldest_turns_after_eight_complete_turns() -> None:
    calls: list[tuple[str, list[ConversationTurn], ConversationSummary | None, str]] = []

    def summarize(
        conversation_id: str,
        turns: list[ConversationTurn],
        existing_summary: ConversationSummary | None,
        trace_id: str,
    ) -> ConversationSummary:
        calls.append((conversation_id, list(turns), existing_summary, trace_id))
        return ConversationSummary(
            version=99,
            covered_until=NOW,
            data={
                "customer_goal": ",".join(turn.customer_text for turn in turns),
                "confirmed_entities": [],
                "customer_preferences": [],
                "unresolved_questions": [],
                "decisions": [],
                "handoff_state": "none",
            },
        )

    policy = RollingSummaryPolicy(max_recent_turns=6, summarize_at_turns=8, summarize_fn=summarize)
    memory = InMemoryConversationMemory(summary_policy=policy)
    for index in range(8):
        memory.append_customer(
            make_event(
                event_key=f"evt-{index}",
                text=f"customer-{index}",
                occurred_at=NOW + timedelta(minutes=index * 2),
            )
        )
        memory.append_assistant(
            make_command(
                dedup_key=f"reply-{index}",
                text=f"assistant-{index}",
                trace_id=f"trace-{index}",
            )
        )

    context = memory.context("conversation-1")

    assert len(calls) == 1
    assert [turn.customer_text for turn in calls[0][1]] == ["customer-0", "customer-1"]
    assert context.summary is not None
    assert context.summary.data["customer_goal"] == "customer-0,customer-1"
    assert [turn.customer_text for turn in context.recent_turns] == [
        "customer-2",
        "customer-3",
        "customer-4",
        "customer-5",
        "customer-6",
        "customer-7",
    ]


def test_memory_falls_back_to_recent_turns_when_summary_generation_fails() -> None:
    def summarize(
        conversation_id: str,
        turns: list[ConversationTurn],
        existing_summary: ConversationSummary | None,
        trace_id: str,
    ) -> ConversationSummary:
        raise RuntimeError("summary failed")

    policy = RollingSummaryPolicy(max_recent_turns=6, summarize_at_turns=8, summarize_fn=summarize)
    memory = InMemoryConversationMemory(summary_policy=policy)
    for index in range(8):
        memory.append_customer(
            make_event(
                event_key=f"evt-{index}",
                text=f"customer-{index}",
                occurred_at=NOW + timedelta(minutes=index * 2),
            )
        )
        memory.append_assistant(
            make_command(
                dedup_key=f"reply-{index}",
                text=f"assistant-{index}",
                trace_id=f"trace-{index}",
            )
        )

    context = memory.context("conversation-1")

    assert context.summary is None
    assert [turn.customer_text for turn in context.recent_turns] == [
        "customer-2",
        "customer-3",
        "customer-4",
        "customer-5",
        "customer-6",
        "customer-7",
    ]


def test_memory_preserves_unmatched_latest_customer_message() -> None:
    memory = InMemoryConversationMemory()
    for index in range(2):
        memory.append_customer(
            make_event(
                event_key=f"evt-{index}",
                text=f"customer-{index}",
                occurred_at=NOW + timedelta(minutes=index * 2),
            )
        )
        memory.append_assistant(
            make_command(
                dedup_key=f"reply-{index}",
                text=f"assistant-{index}",
                trace_id=f"trace-{index}",
            )
        )
    memory.append_customer(
        make_event(
            event_key="evt-2",
            text="customer-2",
            occurred_at=NOW + timedelta(minutes=4),
        )
    )

    context = memory.context("conversation-1")

    assert [turn.customer_text for turn in context.recent_turns] == ["customer-0", "customer-1"]
    assert [turn.assistant_text for turn in context.recent_turns] == ["assistant-0", "assistant-1"]
    assert context.latest_customer_message == "customer-2"


def test_memory_context_prefers_latest_raw_message_over_summary() -> None:
    memory = InMemoryConversationMemory()
    memory.append_customer(make_event(event_key="evt-1", text="latest customer", occurred_at=NOW))
    memory.save_summary("conversation-1", make_summary())

    context = memory.context("conversation-1")

    assert context.summary == make_summary()
    assert context.latest_customer_message == "latest customer"


def test_memory_keeps_raw_customer_text_and_sanitizes_summary_input() -> None:
    captured: list[list[ConversationTurn]] = []

    def summarize(
        conversation_id: str,
        turns: list[ConversationTurn],
        existing_summary: ConversationSummary | None,
        trace_id: str,
    ) -> ConversationSummary:
        del conversation_id, existing_summary, trace_id
        captured.append(list(turns))
        return make_summary()

    memory = InMemoryConversationMemory(
        summary_policy=RollingSummaryPolicy(
            max_recent_turns=0,
            summarize_at_turns=1,
            summarize_fn=summarize,
        )
    )
    secret = "verification code 123456"
    memory.append_customer(make_event(event_key="evt-1", text=secret, occurred_at=NOW))
    memory.append_assistant(make_command(dedup_key="reply-1", text="ok", trace_id="trace-1"))

    context = memory.context("conversation-1")

    assert context.latest_customer_message == secret
    assert captured[0][0].customer_text == "[redacted]"
    assert captured[0][0].assistant_text == "ok"


@pytest.mark.parametrize(
    "secret",
    [
        "482913",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
        "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_conservative_summary_redacts_unlabeled_credential_shaped_values(
    secret: str,
) -> None:
    memory = InMemoryConversationMemory(
        summary_policy=RollingSummaryPolicy(
            max_recent_turns=0,
            summarize_at_turns=1,
            summarize_fn=conservative_structured_summary,
        )
    )
    memory.append_customer(make_event(event_key="evt-1", text=secret, occurred_at=NOW))
    memory.append_assistant(
        make_command(dedup_key="reply-1", text="ok", trace_id="trace-1")
    )

    context = memory.context("conversation-1")

    assert context.latest_customer_message == secret
    assert context.summary is not None
    assert secret not in str(context.summary.data)
    assert context.summary.data["customer_goal"] == "[redacted]"


def test_summary_output_redacts_secrets_from_all_structured_fields() -> None:
    secret = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    secret_top_level_key = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    secret_nested_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue"

    def summarize(*_args) -> ConversationSummary:
        return ConversationSummary(
            version=1,
            covered_until=NOW,
            data={
                "customer_goal": secret,
                "confirmed_entities": [],
                "customer_preferences": [],
                "unresolved_questions": [],
                "decisions": [],
                "handoff_state": "none",
                "constraints": [secret],
                "open_items": {
                    "safe_item": "keep me",
                    secret_nested_key: "drop me",
                    "safe_detail": secret,
                },
                secret_top_level_key: "drop me too",
            },
        )

    memory = InMemoryConversationMemory(
        summary_policy=RollingSummaryPolicy(
            max_recent_turns=0,
            summarize_at_turns=1,
            summarize_fn=summarize,
        )
    )
    memory.append_customer(make_event(event_key="evt-1", text=secret, occurred_at=NOW))
    memory.append_assistant(
        make_command(dedup_key="reply-1", text="ok", trace_id="trace-1")
    )

    summary = memory.context("conversation-1").summary

    assert summary is not None
    assert secret not in str(summary.data)
    assert secret_top_level_key not in summary.data
    assert secret_nested_key not in summary.data["open_items"]
    assert {
        "customer_goal",
        "confirmed_entities",
        "customer_preferences",
        "unresolved_questions",
        "decisions",
        "handoff_state",
    }.issubset(summary.data)
    assert summary.data["constraints"] == ["[redacted]"]
    assert summary.data["open_items"] == {
        "safe_item": "keep me",
        "safe_detail": "[redacted]",
    }


def test_memory_pairs_messages_in_append_order_when_event_times_are_out_of_order() -> None:
    memory = InMemoryConversationMemory()
    memory.append_customer(
        make_event(
            event_key="evt-older",
            text="older customer",
            occurred_at=NOW + timedelta(minutes=2),
        )
    )
    memory.append_customer(
        make_event(
            event_key="evt-newer",
            text="newer customer",
            occurred_at=NOW + timedelta(minutes=1),
        )
    )
    memory.append_assistant(
        make_command(dedup_key="reply-1", text="reply-1", trace_id="trace-1")
    )
    memory.append_assistant(
        make_command(dedup_key="reply-2", text="reply-2", trace_id="trace-2")
    )

    context = memory.context("conversation-1")

    assert [turn.customer_text for turn in context.recent_turns] == ["newer customer"]
    assert [turn.assistant_text for turn in context.recent_turns] == ["reply-1"]
    assert context.latest_customer_message == "newer customer"
