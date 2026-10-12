"""Shared bearer-token job API for browser and mini-program clients.

Run one worker; sessions and jobs are ephemeral.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
import os
import secrets
import time
from pathlib import Path
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from whatsapp_support.config import RagSettings
from whatsapp_support.web_chat import ChatInput, create_app as create_chat_app, prepared_manual_directory


@dataclass
class DemoSession:
    client: httpx.AsyncClient
    touched: float = field(default_factory=time.monotonic)
    jobs: dict = field(default_factory=dict)
    active: bool = False


class Login(BaseModel):
    invite: str = Field(min_length=16, max_length=200)


class Message(ChatInput):
    request_id: str = Field(pattern=r'^[A-Za-z0-9_-]{8,80}$')


def create_app(settings, *, invite, knowledge_dir='knowledge', dataset='sample', public_demo=False):
    if len(invite) < 16:
        raise ValueError('Set a private demo invite of at least 16 characters')
    inner = create_chat_app(settings, knowledge_dir=knowledge_dir, dataset=dataset)
    sessions, tasks = {}, set()
    login_attempts = deque()
    public_message_count = 0

    @asynccontextmanager
    async def lifespan(app):
        async with inner.router.lifespan_context(inner):
            app.state.chat = inner
            yield
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for session in sessions.values():
                await session.client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/healthz')
    async def health():
        return {'status': 'ok'}

    @app.middleware('http')
    async def limits(request, call_next):
        if public_demo and request.url.path == '/demo/session':
            now = time.monotonic()
            while login_attempts and now - login_attempts[0] >= 60:
                login_attempts.popleft()
            if len(login_attempts) >= 10:
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail': '连接请求过多，请一分钟后再试'}, status_code=429,
                                    headers={'Retry-After': '60', 'Cache-Control': 'no-store'})
            login_attempts.append(now)
        if request.method == 'POST':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 16384:
                    raise_error = {'detail': '请求过长'}
                    from fastapi.responses import JSONResponse
                    return JSONResponse(raise_error, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    def session_for(request):
        authorization = request.headers.get('authorization', '')
        token = authorization[7:] if authorization.startswith('Bearer ') else ''
        session = sessions.get(token)
        if session is None or time.monotonic() - session.touched > 3600:
            raise HTTPException(401, '演示会话已过期，请重新连接')
        session.touched = time.monotonic()
        return session

    @app.post('/demo/session')
    async def login(body: Login):
        if not secrets.compare_digest(body.invite.encode(), invite.encode()):
            raise HTTPException(403, '演示口令错误')
        for key, session in list(sessions.items()):
            if not session.active and time.monotonic() - session.touched > 3600:
                await session.client.aclose()
                del sessions[key]
        if len(sessions) >= 16:
            raise HTTPException(429, '演示会话已满')
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=inner),
            base_url='http://localhost', headers={'X-Chat-Client': '1'})
        response = await client.post('/api/session', json={})
        if response.status_code != 200:
            await client.aclose()
            raise HTTPException(503, '无法创建会话')
        token = secrets.token_urlsafe(32)
        sessions[token] = DemoSession(client)
        return {'token': token, **response.json()}

    async def run(session, key, body):
        try:
            response = await asyncio.wait_for(session.client.post('/api/chat',
                json={'text': body.text, 'locale': body.locale}), timeout=180)
            if response.status_code != 200:
                raise ValueError('Chat failed')
            data = response.json()
            session.jobs[key].update(status='done', result={k: data.get(k) for k in
                ('reply', 'status', 'reason', 'sources', 'evidence_count', 'stage')})
        except Exception:
            session.jobs[key].update(status='failed', error='生成失败或超时，请勿反复提交。')
        finally:
            session.active = False

    @app.post('/demo/resume')
    @app.post('/demo/new-conversation')
    async def conversation_control(request: Request):
        session = session_for(request)
        if session.active:
            raise HTTPException(409, '请等待当前回复完成')
        session.active = True
        try:
            path = '/api/resume' if request.url.path.endswith('/resume') else '/api/session'
            response = await session.client.post(path, json={})
            if response.status_code != 200:
                raise HTTPException(response.status_code, '操作未完成，请重新连接或稍后重试')
            return response.json()
        finally:
            session.active = False

    @app.post('/demo/messages', status_code=202)
    async def send(body: Message, request: Request):
        nonlocal public_message_count
        session = session_for(request)
        key = body.request_id
        identity = (body.text, body.locale)
        if key in session.jobs:
            if session.jobs[key]['identity'] != identity:
                raise HTTPException(409, '请求编号已用于另一条消息')
            return {'job_id': key}
        if session.active:
            raise HTTPException(409, '请等待上一条回复')
        if public_demo and (public_message_count >= 100 or len(tasks) >= 2):
            raise HTTPException(429, '公网演示额度已达上限或服务繁忙，请稍后联系演示管理员')
        if len(session.jobs) >= 100:
            raise HTTPException(429, '演示消息已达上限')
        if not body.text.strip():
            raise HTTPException(422, '请输入消息')
        session.active = True
        public_message_count += 1
        session.jobs[key] = {'identity': identity, 'status': 'pending'}
        task = asyncio.create_task(run(session, key, body))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {'job_id': key}

    @app.get('/demo/messages/{key}')
    async def poll(key: str, request: Request):
        job = session_for(request).jobs.get(key)
        if job is None:
            raise HTTPException(404, '没有此任务')
        return {k: v for k, v in job.items() if k != 'identity'}

    # Both interfaces use this exact service instance, index and provider resources.
    # Keep mini routes above the mount. Inner cookies still isolate each session.
    if not public_demo:
        app.mount('/', inner)
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--public-demo', action='store_true', help='Disable web routes and bound public demo usage')
    parser.add_argument('--dataset', choices=['sample', 'emanual'], default='emanual')
    parser.add_argument('--prepared-root', type=Path, default=Path('.rag-data/prepared-public-v3'))
    args = parser.parse_args()
    settings = (RagSettings.from_provider_files('.rag-data/bailian.local.json', '.rag-data/deepseek.local.json')
                if args.live else RagSettings())
    knowledge_dir = prepared_manual_directory(args.prepared_root) if args.dataset == 'emanual' else Path('knowledge')
    settings = replace(settings, index_path=f'.rag-index/web-chat-{args.dataset}-{settings.mode}.sqlite3')
    app = create_app(settings, invite=os.environ.get('WECHAT_DEMO_INVITE', ''),
                     knowledge_dir=str(knowledge_dir), dataset=args.dataset, public_demo=args.public_demo)
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
