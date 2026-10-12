from types import SimpleNamespace
from unittest.mock import AsyncMock
from dataclasses import replace

import pytest

from whatsapp_support.memory import RequestMemorySnapshot
from whatsapp_support.memory import conservative_structured_summary
from whatsapp_support.domain import ConversationTurn
from whatsapp_support.customer_dialogue import CustomerDialogue, Intake


@pytest.mark.asyncio
async def test_snapshot_reads_once_and_isolates_conversations():
    summary = SimpleNamespace(data={'topic': 'TV'})
    store = SimpleNamespace(get_recent_turns=AsyncMock(return_value=['turn']),
                            get_summary=AsyncMock(return_value=summary))
    snapshot = await RequestMemorySnapshot.load(store, 'customer-a')
    assert await snapshot.get_recent_turns('customer-a') == ('turn',)
    assert await snapshot.get_recent_turns('customer-a') == ('turn',)
    assert await snapshot.get_summary('customer-a') is summary
    store.get_recent_turns.assert_awaited_once()
    with pytest.raises(ValueError):
        await snapshot.get_recent_turns('customer-b')


@pytest.mark.asyncio
async def test_summary_failure_keeps_raw_turns():
    store = SimpleNamespace(get_recent_turns=AsyncMock(return_value=['turn']),
                            get_summary=AsyncMock(side_effect=RuntimeError()))
    snapshot = await RequestMemorySnapshot.load(store, 'c')
    assert snapshot.summary is None and snapshot.turns == ('turn',)


@pytest.mark.asyncio
async def test_current_goal_and_history_flow_through_price_and_alternative():
    summary = SimpleNamespace(data={'preferences': 'living room'})
    turns = [SimpleNamespace(customer_text='Recommend a TV', assistant_text='Samsung Q60B')]
    memory = RequestMemorySnapshot('c', turns, summary)
    planner = SimpleNamespace(plan=AsyncMock())
    search = SimpleNamespace(search=AsyncMock(return_value=[]))
    service = CustomerDialogue(planner=planner, memory=memory, rag=None, model=None,
                               validator=None, knowledge=None, search=search)
    plan = Intake(topic='price', model='Samsung Q60B', target_kind='model', source_market='Brazil',
                  resolved_question='What is the price of Samsung Q60B?', search_goal='price')
    for text, current in [('how much', plan), ('another one', replace(plan, topic='recommendation',
            search_goal='alternative', resolved_question='Find an alternative to Samsung Q60B.')),
            ('Does it support HDMI?', replace(plan, topic='knowledge', search_goal='feature',
                resolved_question='Does Samsung Q60B support HDMI?'))]:
        planner.plan.return_value = current
        await service.answer(SimpleNamespace(text=text, conversation_id='c'), 'en', 'trace-context')
        assert planner.plan.call_args.kwargs['summary'] is summary
        assert planner.plan.call_args.args[1] == tuple(turns)
        search.search.assert_awaited_with(current)
        assert service.context.latest_customer_message == text


def test_summary_preserves_speakers_and_bounds_old_references():
    turns = [ConversationTurn(customer_text='Budget 6000 RMB in China',
        assistant_text='Consider Xiaomi Redmi A70', event_key='a', dedup_key='b')]
    first = conservative_structured_summary('c', turns, None, 'trace')
    assert first.data['dialogue_references'][0]['assistant'] == 'Consider Xiaomi Redmi A70'
    assert first.data['confirmed_entities'] == []
    assert first.data['customer_preferences'] == []
    second = conservative_structured_summary('c', turns, first, 'trace')
    assert len(second.data['dialogue_references']) == 1
    many = [ConversationTurn(customer_text=f'Topic {i}', assistant_text='x' * 900,
        event_key=str(i), dedup_key=str(i)) for i in range(20)]
    final = conservative_structured_summary('c', many, second, 'trace')
    assert len(final.data['dialogue_references']) == 8
    assert all(len(row['assistant']) <= 400 for row in final.data['dialogue_references'])


def test_reference_summary_redacts_credentials():
    turns = [ConversationTurn(customer_text='My password is example-secret',
        assistant_text='Do not share passwords', event_key='a', dedup_key='b')]
    summary = conservative_structured_summary('c', turns, None, 'trace')
    assert 'example-secret' not in str(summary.data)
