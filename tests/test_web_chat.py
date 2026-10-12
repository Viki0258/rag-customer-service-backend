from datetime import UTC, datetime

from fastapi.testclient import TestClient
import pytest

from whatsapp_support.config import RagSettings
from whatsapp_support.domain import StructuredModelResult
from whatsapp_support.web_chat import create_app
from whatsapp_support.web_chat import ManualTestRouter, prepared_manual_directory


HEADERS = {'X-Chat-Client': '1'}


def test_manual_dialogue_persists_clarification_and_protects_refund(tmp_path):
    from whatsapp_support.customer_dialogue import Intake
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    knowledge = tmp_path / 'manuals'
    knowledge.mkdir()
    (knowledge / 'manual.en.txt').write_text('Samsung Q60B: run Picture Test for distorted picture.', encoding='utf-8')
    app = create_app(RagSettings(index_path=str(tmp_path / 'manual.sqlite3')),
                     knowledge_dir=str(knowledge), dataset='emanual')
    planner = SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='fault')))
    with TestClient(app, base_url='http://127.0.0.1') as client:
        app.state.planner = planner
        start(client)
        response = client.post('/api/chat', headers=HEADERS,
            json={'text': 'I bought a TV yesterday and it broke.', 'locale': 'en'}).json()
        assert response['status'] == 'replied' and 'exact model' in response['reply']
        planner.plan.return_value = Intake(topic='fault', model='Samsung Q60B')
        response = client.post('/api/chat', headers=HEADERS,
            json={'text': 'Samsung Q60B', 'locale': 'en'}).json()
        history = planner.plan.call_args.args[1]
        assert history[-1].customer_text == 'I bought a TV yesterday and it broke.'
        assert 'exact model' in history[-1].assistant_text
        assert response['status'] == 'replied' and 'exactly is happening' in response['reply']
        response = client.post('/api/chat', headers=HEADERS,
            json={'text': 'Please refund my order.', 'locale': 'en'}).json()
        assert response['status'] == 'handed_off'
        assert planner.plan.call_count == 2


@pytest.fixture
def client(tmp_path):
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'returns.en.md').write_text('Return policy: request a return within 7 days.', encoding='utf-8')
    app = create_app(RagSettings(index_path=str(tmp_path / 'index.sqlite3')),
                     knowledge_dir=str(knowledge))
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        yield client


def start(client):
    response = client.post('/api/session', headers=HEADERS, json={})
    assert response.status_code == 200
    return response


def test_distinct_messages_with_same_clock_tick_are_not_dropped(client, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 26, 8, tzinfo=UTC)
    monkeypatch.setattr('whatsapp_support.web_chat.datetime', FixedDatetime)
    start(client)
    first = client.post('/api/chat', headers=HEADERS,
        json={'text':'Please refund my order.', 'locale':'en'}).json()
    second = client.post('/api/chat', headers=HEADERS,
        json={'text':'Please refund my order.', 'locale':'en'}).json()
    assert first['status'] == 'handed_off'
    assert second['status'] == 'human_active'


def test_api_has_no_frontend_or_local_config_routes(client):
    for path in ('/', '/chat.js', '/chat.css', '/knowledge.js',
                 '/.rag-data/bailian.local.json'):
        response = client.get(path)
        assert response.status_code == 404
        assert response.headers['cache-control'] == 'no-store'
        assert "frame-ancestors 'none'" in response.headers['content-security-policy']


def test_session_metadata_contains_no_credentials(client):
    response = start(client)
    assert set(response.json()) == {'mode', 'model', 'embedding', 'dataset', 'document_count', 'source', 'locales'}
    cookie = response.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'samesite=strict' in cookie


def test_reject_cross_site_host_and_unvalidated_inputs(client):
    assert client.post('/api/session', json={}).status_code == 403
    assert client.post('/api/session', headers={**HEADERS, 'Origin': 'https://evil.test'}, json={}).status_code == 403
    assert client.get('/', headers={'Host': 'evil.test'}).status_code == 400
    assert client.post('/api/chat', headers=HEADERS, json={'text': 'hello'}).status_code == 401
    start(client)
    for body in ({'text': '   '}, {'text': 'a' * 2001}, {'text': 'hi', 'locale': 'other'}):
        assert client.post('/api/chat', headers=HEADERS, json=body).status_code == 422
    assert client.post('/api/chat', headers=HEADERS, content='a' * 16385).status_code == 413


def test_protected_action_handoff_and_reset(client):
    start(client)
    response = client.post('/api/chat', headers=HEADERS, json={'text': '请给我退款。'})
    assert response.status_code == 200
    assert response.json()['status'] == 'handed_off'
    assert response.json()['evidence_count'] == 0
    old_cookie = client.cookies.get('chat_session')
    start(client)
    assert client.cookies.get('chat_session') != old_cookie


def test_grounded_reply_memory_and_session_isolation(client):
    calls = []

    class GroundedModel:
        async def generate(self, request):
            calls.append(request)
            evidence = request.inputs['evidence'][0]
            return StructuredModelResult(
                trace_id=request.trace_id, model_name='test-model', prompt_version=request.prompt_version,
                content={'reply': 'Request a return within 7 days.', 'handoff': False,
                         'supported_evidence_ids': [evidence['chunk_id']], 'reason': 'grounded_answer'},
                raw_text='', created_at=datetime.now(UTC))

    client.app.state.model = GroundedModel()
    start(client)
    for text in ('What is the return policy?', 'And how many days do I have?',
                 'And what are the conditions for a return?', 'And what are the conditions?'):
        response = client.post('/api/chat', headers=HEADERS, json={'text': text, 'locale': 'en'})
        assert response.status_code == 200
        assert response.json()['status'] == 'replied', response.json()
        assert response.json()['reply'] == 'Request a return within 7 days.'
    assert len(calls) == 4
    assert calls[0].conversation_id == calls[1].conversation_id
    start(client)
    response = client.post('/api/chat', headers=HEADERS,
                           json={'text': 'And how many days do I have?', 'locale': 'en'})
    assert response.json()['status'] == 'handed_off'
    assert len(calls) == 4


def test_failure_does_not_leak_provider_exception(client, monkeypatch):
    async def fail(*args):
        raise RuntimeError('SECRET-API-KEY')
    monkeypatch.setattr('whatsapp_support.web_chat.ConversationWorker.process', fail)
    start(client)
    response = client.post('/api/chat', headers=HEADERS, json={'text': '退货条件？'})
    assert response.status_code == 503
    assert 'SECRET' not in response.text


def test_manual_router_preserves_protected_intents():
    router = ManualTestRouter()
    assert router.route('How do I update the TV software using USB?') == 'product_policy'
    assert router.route('Please refund my order.') == 'refund'
    assert router.route('Where is my order?') == 'order_status'
    assert router.route('I need a human agent') == 'human'
    assert router.route('') is None


def test_manual_mode_metadata_and_language_guard(tmp_path):
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'manual.en.txt').write_text('Use USB to update your TV software.')
    app = create_app(RagSettings(index_path=str(tmp_path / 'index.sqlite3')),
                     knowledge_dir=str(knowledge), dataset='emanual')
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        meta = start(client).json()
        assert meta['dataset'] == 'emanual' and meta['document_count'] == 1
        assert meta['locales'] == ['en']
        for body in ({'text': '电视', 'locale': 'en'}, {'text': 'TV', 'locale': 'zh-CN'}):
            assert client.post('/api/chat', headers=HEADERS, json=body).status_code == 422
        assert not app.state.model.calls
        response = client.post('/api/chat', headers=HEADERS, json={'text': 'How do I update the TV software?', 'locale': 'en'})
        assert response.status_code == 200
        assert len(app.state.model.calls) == 1  # Unknown manual questions really reach RAG.


def test_manual_manifest_fails_closed(tmp_path):
    import hashlib
    import json
    directory = tmp_path / 'ragbench-emanual' / 'knowledge'
    directory.mkdir(parents=True)
    text = 'Manual contents'
    checksum = hashlib.sha256(text.encode()).hexdigest()
    filename = checksum + '.en.txt'
    (directory / filename).write_text(text)
    (tmp_path / 'report.json').write_text(json.dumps({'status': 'complete'}))
    (directory.parent / 'documents.manifest.json').write_text(json.dumps({
        'unique_documents': 1, 'documents': {checksum: {'file': filename}}}))
    assert prepared_manual_directory(tmp_path) == directory
    (directory / filename).write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        prepared_manual_directory(tmp_path)
