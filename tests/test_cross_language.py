import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from whatsapp_support.cross_language import CrossLanguageKnowledge, QueryTranslator
from whatsapp_support.domain import RetrievalQuery


@pytest.mark.asyncio
@pytest.mark.parametrize('source,target,translated', [
    ('花屏怎么办', 'en', 'How to fix a distorted picture?'),
    ('How long does delivery take?', 'zh-CN', '配送需要多久？'),
])
async def test_cross_language_retrieves_original_evidence(source, target, translated):
    chunk = SimpleNamespace(chunk_id='original-id', fused_score=.03, text='original evidence')
    knowledge = SimpleNamespace(retrieve=AsyncMock(return_value=[chunk]))
    translator = SimpleNamespace(translate=AsyncMock(return_value=translated))
    embedding = SimpleNamespace(embed=AsyncMock(return_value=[[.1, .2]]))
    service = CrossLanguageKnowledge(knowledge, embedding, translator, [target])
    query = RetrievalQuery(source, 'conversation', 3, locale='zh-CN', query_vector=(1., 0.))
    assert await service.retrieve(query) == [chunk]
    sent = knowledge.retrieve.call_args.args[0]
    assert sent.query_text == translated and sent.locale == target
    assert sent.query_vector == (.1, .2)
    assert query.query_text == source


@pytest.mark.asyncio
async def test_same_language_does_not_translate_or_reembed():
    knowledge = SimpleNamespace(retrieve=AsyncMock(return_value=[]))
    translator = SimpleNamespace(translate=AsyncMock())
    embedding = SimpleNamespace(embed=AsyncMock())
    service = CrossLanguageKnowledge(knowledge, embedding, translator, ['en'])
    await service.retrieve(RetrievalQuery('Shipping time?', 'c', 3, locale='zh-CN'))
    translator.translate.assert_not_called()
    embedding.embed.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('output,valid', [
    ('TV KEEP0END costs KEEP1END RMB', True),
    ('TV Q60C costs 500 RMB', False),
    ('KEEP0END KEEP0END KEEP1END', False),
])
async def test_translation_preserves_model_and_amount(output, valid):
    transport = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=200,
        payload={'output': [{'content': [{'type': 'output_text', 'text': json.dumps({'translation': output})}]}]})))
    settings = SimpleNamespace(response_base_url='https://example.test', response_api_key='fake',
                               response_model='fake', response_timeout_seconds=1)
    translator = QueryTranslator(settings, transport)
    if valid:
        assert await translator.translate('Q60B价格6000元', 'en') == 'TV Q60B costs 6000 RMB'
    else:
        with pytest.raises(ValueError):
            await translator.translate('Q60B价格6000元', 'en')


@pytest.mark.asyncio
async def test_failed_translation_does_not_search_untranslated_fallback():
    knowledge = SimpleNamespace(retrieve=AsyncMock())
    service = CrossLanguageKnowledge(knowledge, None,
        SimpleNamespace(translate=AsyncMock(side_effect=ValueError('unavailable'))), ['en'])
    with pytest.raises(ValueError):
        await service.retrieve(RetrievalQuery('配送多久', 'c', 3, locale='zh-CN'))
    knowledge.retrieve.assert_not_called()


def test_live_wiring_chinese_question_english_corpus(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    from fastapi.testclient import TestClient
    from whatsapp_support.config import RagSettings
    from whatsapp_support.domain import StructuredModelResult
    from whatsapp_support import web_chat

    fake_settings = RagSettings(index_path=str(tmp_path / 'index.sqlite3'))
    settings = SimpleNamespace(**vars(fake_settings))
    settings.mode = 'openai'
    settings.create_embedding = fake_settings.create_embedding
    model = SimpleNamespace(generate=AsyncMock())
    async def generate(request):
        evidence = request.inputs['evidence']
        assert evidence and evidence[0]['locale'] == 'en'
        assert request.inputs['locale'] == 'zh-CN'
        return StructuredModelResult(trace_id=request.trace_id, model_name='test-model',
            prompt_version=request.prompt_version, content={
                'reply': '标准配送需要3到5天。', 'handoff': False,
                'supported_evidence_ids': [evidence[0]['chunk_id']], 'reason': 'grounded_answer'},
            raw_text='', created_at=datetime.now(UTC))
    model.generate.side_effect = generate
    settings.create_model = lambda: model
    translator = SimpleNamespace(translate=AsyncMock(return_value='Standard shipping takes 3 to 5 days.'),
                                 aclose=AsyncMock())
    monkeypatch.setattr(web_chat, 'QueryTranslator', lambda _: translator)
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'shipping.en.md').write_text('Standard shipping takes 3 to 5 days.', encoding='utf-8')
    app = web_chat.create_app(settings, knowledge_dir=str(knowledge))
    with TestClient(app, base_url='http://localhost') as client:
        # This test isolates retrieval translation, not the intake provider.
        app.state.planner = None
        headers = {'X-Chat-Client': '1'}
        assert client.post('/api/session', json={}, headers=headers).status_code == 200
        response = client.post('/api/chat', json={'text': '标准配送要多久？', 'locale': 'zh-CN'}, headers=headers)
        assert response.status_code == 200
        assert response.json()['reply'] == '标准配送需要3到5天。'
        translator.translate.assert_awaited_once()
