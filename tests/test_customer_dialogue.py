from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from whatsapp_support.customer_dialogue import CustomerDialogue, Intake, ModelScopedKnowledge
from whatsapp_support.web_search import DeepSeekSearch


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,model', [('series', 'Samsung Crystal UHD'), ('ambiguous', ''), ('', '')])
async def test_price_without_exact_target_never_searches_for_another_tv(kind, model):
    service = dialogue(Intake(topic='price', model=model, target_kind=kind))
    service.search = SimpleNamespace(search=AsyncMock())
    result = await service.answer(SimpleNamespace(text='how much', conversation_id='c'), 'en', 'trace-price')
    assert result.reason == 'price_target_unresolved' and result.status == 'clarify'
    service.search.search.assert_not_called()


@pytest.mark.asyncio
async def test_exact_price_target_reaches_search_with_market():
    plan = Intake(topic='price', model='Samsung Q60B', target_kind='model', size='70 inches',
                  source_market='Brazil')
    service = dialogue(plan)
    service.search = SimpleNamespace(search=AsyncMock(return_value=[]))
    await service.answer(SimpleNamespace(text='how much', conversation_id='c'), 'en', 'trace-price')
    service.search.search.assert_awaited_once_with(plan)


@pytest.mark.asyncio
async def test_price_evidence_for_different_model_never_reaches_generation():
    service = dialogue(Intake(topic='price', model='Samsung Q60B', target_kind='model'))
    service.search = SimpleNamespace(search=AsyncMock(return_value=[{
        'title':'Samsung Q60C', 'content':'Samsung Q60C costs 500 USD.',
        'url':'https://www.samsung.com/test', 'retrieved_at':'2026-09-23'}]))
    service.model = SimpleNamespace(generate=AsyncMock())
    result = await service.answer(SimpleNamespace(text='how much', conversation_id='c'), 'en', 'trace-price')
    assert result.reason == 'target_evidence_missing'
    service.model.generate.assert_not_called()


def dialogue(plan):
    return CustomerDialogue(planner=SimpleNamespace(plan=AsyncMock(return_value=plan)),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[])),
        rag=SimpleNamespace(answer=AsyncMock()), model=None, validator=None, knowledge=None)


@pytest.mark.asyncio
async def test_fault_asks_instead_of_handoff():
    service = dialogue(Intake(topic='fault'))
    result = await service.answer(SimpleNamespace(text='I bought a TV yesterday. It is broken.', conversation_id='c'), 'en', 't')
    assert result.status == 'clarify'
    assert 'exact model' in result.reply and result.reply.count('?') == 1
    service.rag.answer.assert_not_called()


@pytest.mark.asyncio
async def test_danger_bypasses_model():
    service = dialogue(Intake(topic='fault'))
    result = await service.answer(SimpleNamespace(text='TV is smoking', conversation_id='c'), 'en', 't')
    assert result.reason == 'electrical_safety_handoff'
    service.planner.plan.assert_not_called()


@pytest.mark.asyncio
async def test_missing_search_does_not_invent_recommendation():
    service = dialogue(Intake(topic='recommendation', budget='1000 USD', size='75 inches',
                             market='US', distance='3m', usage='movies'))
    result = await service.answer(SimpleNamespace(text='recommend a TV', conversation_id='c'), 'en', 't')
    assert result.status == 'clarify' and result.reason == 'search_not_configured'
    assert service.sources == []


@pytest.mark.asyncio
async def test_model_scope_does_not_accept_other_models():
    chunks = [SimpleNamespace(text='Samsung Q60B Picture Test'), SimpleNamespace(text='Samsung Q60C Picture Test')]
    knowledge = SimpleNamespace(retrieve=AsyncMock(return_value=chunks))
    assert await ModelScopedKnowledge(knowledge, 'Samsung Q60B').retrieve(None) == chunks[:1]
    assert await ModelScopedKnowledge(knowledge, 'Samsung').retrieve(None) == []


def search_payload(url='https://www.samsung.com/us/tvs/test/', cited=True):
    return {'content': [
        {'type': 'web_search_tool_result', 'content': [{'type': 'web_search_result', 'url': url, 'title': 'TV'}]},
        {'type': 'text', 'text': 'Do not use this generated prose as evidence.',
         'citations': [{'url': url, 'cited_text': '75 inch QLED'}] if cited else []}]}


def test_search_requires_native_results_and_cited_excerpt():
    with pytest.raises(ValueError):
        DeepSeekSearch.parse({'content': [{'type': 'text', 'text': 'I searched the web'}]})
    assert DeepSeekSearch.parse(search_payload(cited=False)) == []
    hits = DeepSeekSearch.parse(search_payload())
    assert len(hits) == 1 and hits[0]['content'] == '75 inch QLED'


@pytest.mark.parametrize('url', ['http://samsung.com/', 'https://samsung.com.evil.test/',
                               'https://evil.test/', 'https://user:pass@samsung.com/', 'https://127.0.0.1/'])
def test_search_rejects_unsafe_or_nonmanufacturer_urls(url):
    assert DeepSeekSearch.parse(search_payload(url)) == []


@pytest.mark.asyncio
async def test_china_6000_70_inches_searches_without_optional_questions():
    service = dialogue(Intake(topic='recommendation', budget='6000 RMB', size='70 inches', market='China'))
    service.search = SimpleNamespace(search=AsyncMock(return_value=[]))
    result = await service.answer(SimpleNamespace(text='China, around 6000 RMB', conversation_id='c'), 'en', 't')
    service.search.search.assert_awaited_once()
    assert result.reason == 'no_verified_web_evidence'
    assert '?' not in result.reply


@pytest.mark.asyncio
@pytest.mark.parametrize('preferences', [{}, {'size': '70 inches'}, {'budget': '6000 RMB'},
                                       {'market': 'China', 'size': '70 inches'}])
async def test_recommendation_searches_immediately_with_partial_or_no_preferences(preferences):
    service = dialogue(Intake(topic='recommendation', **preferences))
    service.search = SimpleNamespace(search=AsyncMock(return_value=[]))
    result = await service.answer(SimpleNamespace(text='recommend a TV', conversation_id='c'), 'en', 't')
    service.search.search.assert_awaited_once_with(Intake(topic='recommendation', **preferences))
    assert '?' not in result.reply
    assert result.reason == 'no_verified_web_evidence'


@pytest.mark.asyncio
async def test_old_multi_question_session_does_not_restart_intake():
    service = dialogue(Intake(topic='recommendation', size='70 inches'))
    service.memory.get_recent_turns.return_value = [SimpleNamespace(
        assistant_text='What is your approximate budget, including the currency? In which country or region will you buy the TV? No address is needed.')]
    result = await service.answer(SimpleNamespace(text='Not sure', conversation_id='c'), 'en', 't')
    assert result.reason == 'search_not_configured'


@pytest.mark.asyncio
async def test_skip_questions_does_not_invent_preferences():
    service = dialogue(Intake(topic='recommendation', size='70 inches'))
    service.search = SimpleNamespace(search=AsyncMock(return_value=[]))
    await service.answer(SimpleNamespace(text='Just recommend one, no more questions', conversation_id='c'), 'en', 't')
    plan = service.search.search.call_args.args[0]
    assert plan.budget == plan.market == plan.distance == plan.usage == ''


@pytest.mark.asyncio
async def test_direct_recommendation_generates_grounded_answer_without_intake():
    from datetime import UTC, datetime
    from whatsapp_support.domain import StructuredModelResult
    from whatsapp_support.rag import RagReplyValidator
    service = dialogue(Intake(topic='recommendation', size='70 inches'))
    service.search = SimpleNamespace(search=AsyncMock(return_value=[{
        'url': 'https://www.samsung.com/us/test/', 'title': 'Test product',
        'retrieved_at': '2026-09-22', 'content': 'Test fixture: Example TV is 70 inches.'}]))
    service.model = SimpleNamespace(generate=AsyncMock(return_value=StructuredModelResult(
        trace_id='trace-recommendation-test', model_name='test', prompt_version='web-recommendation-v1',
        content={'reply': 'Samsung Example TV fits the requested 70-inch size.', 'handoff': False,
                 'supported_evidence_ids': ['web-0'], 'reason': 'grounded_answer'},
        raw_text='', created_at=datetime.now(UTC))))
    service.validator = RagReplyValidator()
    result = await service.answer(SimpleNamespace(text='recommend a 70 inch TV', conversation_id='c'), 'en', 'trace-recommendation-test')
    assert result.status == 'reply' and len(service.sources) == 1
    prompt = service.model.generate.call_args.args[0].inputs['messages'][0]['content']
    assert 'single-model suggestion now' in prompt
    assert 'do not imply local availability' in prompt
