from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from whatsapp_support.customer_dialogue import CustomerDialogue, Intake
from whatsapp_support.domain import StructuredModelResult
from whatsapp_support.rag import RagReplyValidator
from whatsapp_support.web_search import DeepSeekSearch, samsung_source


@pytest.mark.parametrize('url,allowed', [
    ('https://www.samsung.com/cn/tvs/', True),
    ('https://samsung.com/us/tvs/', True),
    ('https://samsung.com.evil.test/tvs/', False),
    ('https://www.tcl.com/cn/', False),
    ('http://www.samsung.com/', False),
    ('https://news.samsung.com/mx/tv-launch', False),
    ('https://www.samsung.com/cn/newsroom/tv-launch', False),
])
def test_official_brand_boundary(url, allowed):
    assert samsung_source(url) is allowed


@pytest.mark.asyncio
async def test_search_filters_competitors_before_fetch():
    transport = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=200,
        payload={'content': [{'type': 'web_search_tool_result', 'content': [
            {'type': 'web_search_result', 'url': 'https://www.tcl.com/cn/tvs/', 'title': 'TCL TV'},
            {'type': 'web_search_result', 'url': 'https://www.samsung.com/cn/tvs/', 'title': 'Samsung TV'},
        ]}]})))
    settings = SimpleNamespace(response_base_url='https://api.deepseek.com', response_api_key='fake',
                               response_model='fake', response_timeout_seconds=1)
    fetch = AsyncMock(return_value='Samsung TV specifications')
    search = DeepSeekSearch(settings, transport, fetch)
    hits = await search.search(Intake(topic='recommendation', market='China'))
    assert len(hits) == 1 and samsung_source(hits[0]['url'])
    fetch.assert_awaited_once_with('https://www.samsung.com/cn/tvs/')
    assert 'ONLY Samsung' in transport.post.call_args.kwargs['json_body']['system']
    assert '75-inch and 85-inch' in transport.post.call_args.kwargs['json_body']['messages'][0]['content']


@pytest.mark.asyncio
@pytest.mark.parametrize('reply,expected', [
    ('推荐三星Q60B。', 'reply'),
    ('推荐TCL 85T7H。', 'clarify'),
    ('相比三星，推荐海信电视。', 'clarify'),
])
async def test_output_cannot_switch_to_competitor(reply, expected):
    service = CustomerDialogue(
        planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='recommendation'))),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None)),
        rag=None, knowledge=None, validator=RagReplyValidator(),
        search=SimpleNamespace(search=AsyncMock(return_value=[{
            'url': 'https://www.samsung.com/cn/tvs/', 'title': 'Samsung Q60B',
            'content': 'Samsung Q60B TV', 'retrieved_at': '2026-09-23'}])),
        model=SimpleNamespace(generate=AsyncMock(return_value=StructuredModelResult(
            trace_id='trace', model_name='fake', prompt_version='v1',
            content={'reply': reply, 'supported_evidence_ids': ['web-0'], 'handoff': False, 'reason': 'grounded_answer'},
            raw_text='', created_at=datetime.now(UTC)))))
    result = await service.answer(SimpleNamespace(text='推荐一款电视', conversation_id='c'), 'zh-CN', 'trace')
    assert result.status == expected
    if expected != 'reply':
        assert result.reason == 'brand_scope_unverified' and not service.sources
