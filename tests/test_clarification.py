from datetime import UTC, datetime

import pytest

from whatsapp_support.clarification import QUESTIONS, clarification_choice, needs_return_time_clarification
from whatsapp_support.domain import ConversationTurn, RetrievedChunk
from whatsapp_support.fakes import FakeConversationMemory, FakeEmbedding, FakeHybridKnowledge, FakeModel
from whatsapp_support.rag import RagAnswerService
from test_policy_followups import event, turn


def pending():
    return [turn('退货有哪些条件？'), ConversationTurn('那要几天？', QUESTIONS['zh'], 'e2', 'r2')]


@pytest.mark.parametrize(('text', 'choice'), [
    ('申请期限', 'deadline'), ('第一个', 'deadline'), ('the deadline', 'deadline'),
    ('办理时长', 'processing'), ('the second one', 'processing'),
    ('不知道', 'unresolved'), ('那要几天？', 'unresolved'),
    ('退款', None), ('转人工', None), ('What is standard shipping?', None),
])
def test_clarification_choices_and_safety(text, choice):
    assert clarification_choice(text, pending()) == choice


def test_customer_cannot_spoof_a_pending_clarification():
    assert clarification_choice('第一个', [turn(QUESTIONS['zh'])]) is None
    assert needs_return_time_clarification('那要几天？', [turn('退货有哪些条件？')])
    assert not needs_return_time_clarification('那要几天？', pending())
    assert not needs_return_time_clarification('How long does it take?', [turn('standard shipping')])


def service(turns, *, evidence=True, memory=None):
    embedding, knowledge, model = FakeEmbedding(), FakeHybridKnowledge(), FakeModel()
    if evidence:
        knowledge.results_queue.append([RetrievedChunk(
            chunk_id='c1', document_id='d1', locale='zh-CN', text='签收后7天内可申请退货。',
            vector_rank=1, keyword_rank=1, fused_score=0.1)])
    rag = RagAnswerService(embedding=embedding, knowledge=knowledge, model=model,
        memory=memory or FakeConversationMemory(recent_turns=turns), prompt_version='test',
        retrieval_top_k=5, evidence_threshold=0, max_retrieved_chunks=5,
        max_reply_chars=500, clock=lambda: datetime.now(UTC))
    return rag, embedding, knowledge, model


@pytest.mark.asyncio
async def test_ambiguity_produces_fixed_clarification_without_model_or_fact_claims():
    rag, _, _, model = service([turn('退货有哪些条件？')])
    result = await rag.answer(event('那要几天？'), 'zh-CN', 'trace')
    assert result.status == 'clarify'
    assert result.reply == QUESTIONS['zh']
    assert result.supported_chunk_ids == ()
    assert model.calls == []


@pytest.mark.asyncio
async def test_no_knowledge_does_not_trigger_clarification():
    rag, _, _, model = service([turn('退货有哪些条件？')], evidence=False)
    result = await rag.answer(event('那要几天？'), 'zh-CN', 'trace')
    assert result.status == 'handoff'
    assert result.reason == 'insufficient_evidence'
    assert not model.calls


@pytest.mark.asyncio
async def test_second_unclear_answer_hands_off_without_another_question():
    rag, embedding, _, model = service(pending())
    incoming = event('不知道')
    assert await rag.resolve_followup_intent(incoming) == 'product_policy'
    result = await rag.answer(incoming, 'zh-CN', 'trace')
    assert result.status == 'handoff'
    assert result.reason == 'clarification_requires_human'
    assert not embedding.calls and not model.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(('choice', 'query'), [('申请期限', '退货申请期限是多久？'),
                                            ('办理时长', '办理退货需要多久？')])
async def test_selection_retrieves_fresh_evidence(choice, query):
    rag, embedding, knowledge, model = service(pending(), evidence=False)
    incoming = event(choice)
    assert await rag.resolve_followup_intent(incoming) == 'product_policy'
    result = await rag.answer(incoming, 'zh-CN', 'trace')
    assert embedding.calls == [[query]]
    assert knowledge.calls[0].query_text == query
    assert result.status == 'handoff'
    assert not model.calls


@pytest.mark.asyncio
async def test_worker_persists_clarification_deduplicates_and_hands_off_next_unclear_turn():
    from dataclasses import replace
    from whatsapp_support.domain import OutboundCommand, OutboundMode
    from whatsapp_support.memory import InMemoryConversationMemory, AsyncInMemoryConversationMemory
    from whatsapp_support.fakes import (InMemoryConversationStore, InMemoryOutboxStore,
        InMemoryConversationOutboxTransaction, FakeIdentity, FakeHandoff, FakeTemplateCatalog)
    from whatsapp_support.worker import ConversationWorker
    memory = InMemoryConversationMemory()
    memory.append_customer(event('退货有哪些条件？'))
    memory.append_assistant(OutboundCommand('r1', 'conversation', 'recipient', OutboundMode.FREEFORM,
        '签收后7天内可申请退货。', None, None, {}, 'allowed', 'seed'))
    rag, _, _, model = service([], memory=AsyncInMemoryConversationMemory(memory))
    conversations, outbox = InMemoryConversationStore(), InMemoryOutboxStore()
    transaction = InMemoryConversationOutboxTransaction(conversations, outbox, memory)
    worker = ConversationWorker(conversations, outbox, None, FakeIdentity(), None, None,
        FakeTemplateCatalog(), FakeHandoff(), rag_answer=rag, transaction=transaction, locale='zh-CN')
    incoming = replace(event('那要几天？'), event_key='clarify-event')
    first = await worker.process(incoming)
    assert first.status == 'replied'
    records = await outbox.records()
    assert len(records) == 1 and records[0].command.text == QUESTIONS['zh']
    assert memory.context('conversation').recent_turns[-1].assistant_text == QUESTIONS['zh']
    await worker.process(incoming)
    assert len(await outbox.records()) == 1
    from datetime import timedelta
    next_event = replace(event('不知道'), event_key='unknown-event',
                         occurred_at=incoming.occurred_at + timedelta(seconds=1),
                         received_at=incoming.received_at + timedelta(seconds=1))
    result = await worker.process(next_event)
    assert result.status == 'handed_off'
    assert len(await outbox.records()) == 1
    assert not model.calls


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['退款', '转人工', 'Where is my order?'])
async def test_protected_request_during_clarification_still_bypasses_rag(text):
    from whatsapp_support.fakes import (InMemoryConversationStore, InMemoryOutboxStore,
        FakeIdentity, FakeHandoff, FakeTemplateCatalog)
    from whatsapp_support.worker import ConversationWorker
    rag, embedding, _, model = service(pending())
    worker = ConversationWorker(InMemoryConversationStore(), InMemoryOutboxStore(), None,
        FakeIdentity(), None, None, FakeTemplateCatalog(), FakeHandoff(), rag_answer=rag)
    result = await worker.process(event(text))
    assert result.status == 'handed_off'
    assert not embedding.calls and not model.calls


@pytest.mark.asyncio
async def test_legacy_worker_without_atomic_history_does_not_start_clarification_loop():
    from whatsapp_support.fakes import (InMemoryConversationStore, InMemoryOutboxStore,
        FakeIdentity, FakeHandoff, FakeTemplateCatalog)
    from whatsapp_support.worker import ConversationWorker
    rag, _, _, _ = service([turn('退货有哪些条件？')])
    outbox = InMemoryOutboxStore()
    worker = ConversationWorker(InMemoryConversationStore(), outbox, None,
        FakeIdentity(), None, None, FakeTemplateCatalog(), FakeHandoff(), rag_answer=rag)
    result = await worker.process(event('那要几天？'))
    assert result.status == 'handed_off'
    assert result.reason == 'clarification_requires_transaction'
    assert not await outbox.records()
