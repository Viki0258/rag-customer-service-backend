"""The mini transport must not select a reduced customer-service pipeline."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
import time

from fastapi.testclient import TestClient
import pytest

from whatsapp_support.config import RagSettings
from whatsapp_support.customer_dialogue import Intake
from whatsapp_support.domain import StructuredModelResult
from whatsapp_support.mini_app import create_app


@pytest.mark.parametrize('dataset', ['sample', 'emanual'])
def test_same_dialogue_through_web_and_mini(tmp_path, dataset):
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'tv.en.md').write_text('Samsung Q60B: use Picture Test for distorted picture.', encoding='utf-8')
    app = create_app(RagSettings(index_path=str(tmp_path / 'index.sqlite3')),
        invite='local-test-invite-1234', knowledge_dir=str(knowledge), dataset=dataset)
    with TestClient(app, base_url='http://localhost') as client:
        shared = app.state.chat
        planner = SimpleNamespace(plan=AsyncMock())
        shared.state.planner = planner
        search = SimpleNamespace(search=AsyncMock(return_value=[{
            'title': 'Samsung Q80', 'content': 'Samsung Q80 has an 80 inch display.',
            'url': 'https://www.samsung.com/test', 'retrieved_at': '2026-09-23'}]))
        shared.state.search = search
        async def generate(request):
            return StructuredModelResult(trace_id=request.trace_id, model_name='test',
                prompt_version=request.prompt_version, content={
                    'reply': 'Consider the Samsung Q80 with an 80 inch display.',
                    'supported_evidence_ids': ['web-0'], 'handoff': False, 'reason': 'grounded_answer'},
                raw_text='', created_at=datetime.now(UTC))
        shared.state.model = SimpleNamespace(generate=generate)
        results = []
        for channel in ['web', 'mini']:
            if channel == 'web':
                metadata = client.post('/api/session', headers={'X-Chat-Client': '1'}, json={}).json()
            else:
                metadata = client.post('/demo/session', json={'invite': 'local-test-invite-1234'}).json()
                auth = {'Authorization': 'Bearer ' + metadata['token']}
            assert metadata['dataset'] == dataset and metadata['document_count'] == 1
            def send(text, seq):
                body = {'text': text, 'locale': 'en'}
                if channel == 'web':
                    return client.post('/api/chat', headers={'X-Chat-Client': '1'}, json=body).json()
                body['request_id'] = 'parity-test-' + str(seq)
                job = client.post('/demo/messages', headers=auth, json=body).json()['job_id']
                for _ in range(200):
                    result = client.get('/demo/messages/' + job, headers=auth).json()
                    if result['status'] != 'pending':
                        assert result['status'] == 'done'
                        return result['result']
                    time.sleep(.01)
                pytest.fail('job did not finish')
            planner.plan.return_value = Intake(topic='recommendation', size='80 inches')
            recommendation = send('Recommend an 80 inch TV for my room.', 1)
            assert recommendation['status'] == 'replied'
            assert recommendation['sources']
            planner.plan.return_value = Intake(topic='price', model='Samsung Q80', target_kind='model')
            send('How much?', 2)
            history = planner.plan.call_args.args[1]
            assert history[-1].customer_text == 'Recommend an 80 inch TV for my room.'
            assert 'Q80' in history[-1].assistant_text
            planner.plan.return_value = Intake(topic='fault')
            fault = send('My TV broke yesterday.', 3)
            assert 'exact model' in fault['reply']
            count = planner.plan.call_count
            refund = send('Please refund my order.', 4)
            assert refund['status'] == 'handed_off'
            assert planner.plan.call_count == count
            results.append((recommendation['reply'], fault['reply'], refund['reason']))
        assert results[0] == results[1]
        assert search.search.call_count == 4


@pytest.mark.asyncio
async def test_recommendation_insufficient_evidence_is_not_human_handoff():
    from whatsapp_support.customer_dialogue import CustomerDialogue
    from whatsapp_support.rag import RagReplyValidator
    planner = SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='recommendation')))
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None))
    model = SimpleNamespace(generate=AsyncMock(return_value=StructuredModelResult(
        trace_id='trace', model_name='test', prompt_version='v1',
        content={'reply': '', 'handoff': True, 'supported_evidence_ids': ['web-0'], 'reason': 'insufficient_evidence'},
        raw_text='', created_at=datetime.now(UTC))))
    search = SimpleNamespace(search=AsyncMock(return_value=[{'title': 'TV', 'content': 'Incomplete specs',
        'url': 'https://www.samsung.com/test', 'retrieved_at': '2026-09-23'}]))
    service = CustomerDialogue(planner=planner, memory=memory, rag=None, model=model,
        validator=RagReplyValidator(), knowledge=None, search=search)
    result = await service.answer(SimpleNamespace(text='推荐电视', conversation_id='c'), 'zh-CN', 'trace')
    assert result.status == 'clarify'
    assert result.reason == 'recommendation_evidence_insufficient'
    assert '无法' in result.reply or '不足' in result.reply
