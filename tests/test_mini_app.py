import time

from fastapi.testclient import TestClient
import pytest

from whatsapp_support.config import RagSettings
from whatsapp_support.mini_app import create_app


INVITE = 'local-test-invite-1234'


@pytest.fixture
def client(tmp_path):
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'shipping.en.md').write_text('Standard shipping takes 3 to 5 days.', encoding='utf-8')
    app = create_app(RagSettings(index_path=str(tmp_path / 'index.sqlite3')),
                     invite=INVITE, knowledge_dir=str(knowledge))
    with TestClient(app) as client:
        yield client


def login(client):
    response = client.post('/demo/session', json={'invite': INVITE})
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    return {'Authorization': 'Bearer ' + response.json()['token']}


def test_private_access(client):
    assert client.get('/healthz').json() == {'status': 'ok'}
    assert client.post('/demo/session', json={'invite': 'wrong-invite-123456'}).status_code == 403
    assert client.get('/demo/messages/missing').status_code == 401
    headers = login(client)
    raw = headers['Authorization'][7:]
    assert client.get('/demo/messages/missing', headers={'Authorization': raw}).status_code == 401
    assert client.get('/demo/messages/missing', headers=headers).status_code == 404


def test_job_roundtrip_idempotency_and_ownership(client):
    headers = login(client)
    body = {'request_id': 'example-1234', 'text': 'How long does shipping take?', 'locale': 'en'}
    sent = client.post('/demo/messages', headers=headers, json=body)
    assert sent.status_code == 202
    assert client.post('/demo/messages', headers=headers, json=body).json() == sent.json()
    assert client.post('/demo/messages', headers=headers, json={**body, 'text': 'Other question'}).status_code == 409
    path = '/demo/messages/' + sent.json()['job_id']
    assert client.get(path, headers=login(client)).status_code == 404
    for _ in range(100):
        job = client.get(path, headers=headers).json()
        if job['status'] != 'pending':
            break
        time.sleep(.01)
    assert job['status'] == 'done'
    assert 'identity' not in job
    assert job['result']['status'] in ('replied', 'handed_off')


def test_input_limits(client):
    headers = login(client)
    assert client.post('/demo/messages', headers=headers, json={
        'request_id': 'example-1234', 'text': ' ', 'locale': 'en'}).status_code == 422
    assert client.post('/demo/session', content=b'x' * 17000).status_code == 413


def test_missing_invite_fails_closed():
    with pytest.raises(ValueError):
        create_app(RagSettings(), invite='')


def test_public_demo_closes_web_bypass_and_limits_login(tmp_path):
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    (knowledge / 'test.en.md').write_text('Test policy.', encoding='utf-8')
    app = create_app(RagSettings(index_path=str(tmp_path / 'index.sqlite3')),
        invite=INVITE, knowledge_dir=str(knowledge), public_demo=True)
    with TestClient(app) as client:
        for path in ('/', '/api/session', '/api/chat', '/api/resume', '/docs'):
            response = client.post(path, json={}) if path.startswith('/api/') else client.get(path)
            assert response.status_code == 404
        assert client.get('/demo/messages/private').status_code == 401
        for _ in range(10):
            assert client.post('/demo/session', json={'invite': 'incorrect-invite-123'}).status_code == 403
        assert client.post('/demo/session', json={'invite': INVITE}).status_code == 429


def test_explicit_resume_and_new_conversation(client):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from whatsapp_support.customer_dialogue import Intake
    planner = SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='fault')))
    client.app.state.chat.state.planner = planner
    headers = login(client)
    def send(text, number):
        key = 'resume-test-' + str(number)
        assert client.post('/demo/messages', headers=headers, json={
            'request_id': key, 'text': text, 'locale': 'en'}).status_code == 202
        for _ in range(100):
            job = client.get('/demo/messages/' + key, headers=headers).json()
            if job['status'] != 'pending':
                return job['result']
            time.sleep(.01)
        pytest.fail('job timed out')
    first = send('My TV broke yesterday.', 1)
    assert 'exact model' in first['reply']
    assert send('Please refund my order.', 2)['status'] == 'handed_off'
    count = planner.plan.call_count
    assert send('Samsung TV has a distorted picture.', 3)['status'] == 'human_active'
    assert planner.plan.call_count == count
    assert client.post('/demo/resume').status_code == 401
    # Another session cannot resume this one.
    other = login(client)
    assert client.post('/demo/resume', headers=other).status_code == 200
    assert send('Samsung TV has a distorted picture.', 4)['status'] == 'human_active'
    assert client.post('/demo/resume', headers=headers).status_code == 200
    planner.plan.return_value = Intake(topic='price')
    send('How much?', 5)
    assert any(turn.customer_text == 'My TV broke yesterday.'
               for turn in planner.plan.call_args.args[1])
    assert send('Please refund my order.', 6)['status'] == 'handed_off'
    assert client.post('/demo/new-conversation', headers=headers).status_code == 200
    planner.plan.return_value = Intake(topic='fault')
    assert 'exact model' in send('My Samsung TV has a distorted picture.', 7)['reply']
    assert not planner.plan.call_args.args[1]
