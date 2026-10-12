"""Shared synchronous chat API used by browser and mini-program transports."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import hashlib
import json
import secrets
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from whatsapp_support.config import RagSettings
from whatsapp_support.customer_dialogue import CustomerDialogue, ModelIntake
from whatsapp_support.web_search import DeepSeekSearch
from whatsapp_support.domain import AutomationState, EventType, InboundEvent
from whatsapp_support.fakes import (
    FakeHandoff, InMemoryConversationOutboxTransaction, InMemoryConversationStore, InMemoryOutboxStore,
)
from whatsapp_support.memory import (
    AsyncInMemoryConversationMemory, InMemoryConversationMemory,
    RollingSummaryPolicy, conservative_structured_summary,
    RequestMemorySnapshot,
)
from whatsapp_support.rag import RagAnswerService
from whatsapp_support.manual_scope import GENERAL_MANUAL_CHECKSUMS
from whatsapp_support.cross_language import CrossLanguageKnowledge, QueryTranslator
from whatsapp_support.reply_diagnostics import DiagnosticReplyValidator
from whatsapp_support.rag_index import SqliteHybridKnowledge, SqliteKnowledgeIndex
from whatsapp_support.worker import ConversationWorker
from whatsapp_support.policy import IntentRouter
from whatsapp_support.rag_index import KnowledgeFileLoader


class ManualTestRouter(IntentRouter):
    """Local manual test only; never overrides known protected business intents."""

    def route(self, text):
        return super().route(text) or ('product_policy' if isinstance(text, str) and text.strip() else None)


def prepared_manual_directory(root: Path) -> Path:
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Public data preparation is incomplete')
    directory = root / 'ragbench-emanual' / 'knowledge'
    manifest = json.loads((directory.parent / 'documents.manifest.json').read_text(encoding='utf-8'))
    expected = set()
    for checksum, record in manifest['documents'].items():
        filename = checksum + '.en.txt'
        if record['file'] != filename or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Invalid manual manifest')
        path = directory / filename
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('Manual document checksum mismatch')
        expected.add(filename)
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
    if not expected or actual != expected or len(expected) != manifest['unique_documents']:
        raise ValueError('Manual document manifest mismatch')
    return directory


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    locale: Literal['zh-CN', 'en'] = 'zh-CN'


@dataclass
class Session:
    memory: InMemoryConversationMemory
    conversations: InMemoryConversationStore = field(default_factory=InMemoryConversationStore)
    outbox: InMemoryOutboxStore = field(default_factory=InMemoryOutboxStore)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    conversation_id: str = field(default_factory=lambda: uuid4().hex)
    touched: float = field(default_factory=time.monotonic)
    turns: int = 0
    last_event_at: datetime | None = None


def create_app(settings: RagSettings, *, knowledge_dir: str = 'knowledge', dataset: str = 'sample') -> FastAPI:
    if dataset not in {'sample', 'emanual'}:
        raise ValueError('Unknown web dataset')
    sessions: dict[str, Session] = {}

    @asynccontextmanager
    async def lifespan(app):
        resources = []
        try:
            embedding = settings.create_embedding()
            resources.append(embedding)
            model = settings.create_model()
            resources.append(model)
            index = SqliteKnowledgeIndex(settings.index_path)
            resources.append(index)
            await index.open()
            documents = KnowledgeFileLoader.load_directory(knowledge_dir)
            app.state.document_count = len(documents)
            app.state.general_manual_ids = frozenset(d.document_id for d in documents
                if dataset == 'emanual' and d.checksum in GENERAL_MANUAL_CHECKSUMS)
            # Dedicated web index; unchanged document embeddings are cached.
            await index.import_directory(knowledge_dir, embedding)
            app.state.embedding, app.state.model = embedding, model
            app.state.planner, app.state.search = None, None
            if settings.mode != 'fake':
                app.state.planner = ModelIntake(settings)
                resources.append(app.state.planner)
                if settings.response_base_url.rstrip('/') == 'https://api.deepseek.com':
                    app.state.search = DeepSeekSearch(settings)
                    resources.append(app.state.search)
            app.state.knowledge = SqliteHybridKnowledge(
                index, vector_candidate_k=settings.vector_candidate_count,
                bm25_candidate_k=settings.bm25_candidate_count,
                final_k=settings.final_candidate_count, threshold=settings.evidence_threshold)
            if settings.mode != 'fake':
                translator = QueryTranslator(settings)
                resources.append(translator)
                app.state.knowledge = CrossLanguageKnowledge(
                    app.state.knowledge, embedding, translator,
                    [d.metadata.get('locale', 'en') for d in documents
                     if d.metadata.get('locale', 'en').split('-')[0] in {'en', 'zh'}])
            yield
        finally:
            sessions.clear()
            for resource in reversed(resources):
                close = getattr(resource, 'aclose', None) or getattr(resource, 'close', None)
                if close:
                    await close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])

    @app.middleware('http')
    async def local_security(request: Request, call_next):
        if request.method == 'POST':
            origin = request.headers.get('origin')
            expected = f'{request.url.scheme}://{request.headers.get("host", "")}'
            if (request.headers.get('x-chat-client') != '1'
                    or (origin is not None and origin != expected)):
                return JSONResponse({'detail': '禁止跨站请求'}, status_code=403)
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 16384:
                    return JSONResponse({'detail': '消息过长'}, status_code=413)
            request._body = bytes(data)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response

    @app.post('/api/session')
    async def new_session(request: Request):
        old_key = request.cookies.get('chat_session', '')
        old = sessions.get(old_key)
        if old and old.lock.locked():
            raise HTTPException(409, '请等待当前回复完成')
        sessions.pop(old_key, None)
        for key, session in list(sessions.items()):
            if not session.lock.locked() and time.monotonic() - session.touched > 3600:
                sessions.pop(key)
        if len(sessions) >= 32:
            raise HTTPException(429, '测试会话已满，请稍后再试')
        key = secrets.token_urlsafe(32)
        sessions[key] = Session(memory=InMemoryConversationMemory(summary_policy=RollingSummaryPolicy(
            max_recent_turns=settings.summary_retention_turns,
            summarize_at_turns=settings.summary_trigger_turns,
            summarize_fn=conservative_structured_summary)))
        response = JSONResponse({'mode': settings.mode, 'model': settings.response_model,
                                 'embedding': settings.embedding_model,
                                 'dataset': dataset, 'document_count': app.state.document_count,
                                 'source': 'Galileo RAGBench · eManual test · CC BY 4.0' if dataset == 'emanual' else 'Synthetic policy fixtures',
                                 'locales': ['en'] if dataset == 'emanual' and settings.mode == 'fake' else ['zh-CN', 'en']})
        response.set_cookie('chat_session', key, httponly=True, samesite='strict', max_age=3600)
        return response

    @app.post('/api/resume')
    async def resume_demo(request: Request):
        # This app is a loopback demo using FakeHandoff, never a production takeover API.
        session = sessions.get(request.cookies.get('chat_session', ''))
        if session is None or time.monotonic() - session.touched > 3600:
            raise HTTPException(401, '会话已过期，请新建会话')
        if session.lock.locked():
            raise HTTPException(409, '请等待当前回复完成')
        async with session.lock:
            conversation = await session.conversations.get(session.conversation_id)
            if conversation and conversation.automation_state == AutomationState.HUMAN_ACTIVE:
                await session.conversations.save(replace(conversation,
                    automation_state=AutomationState.BOT_ACTIVE, version=conversation.version + 1))
            session.touched = time.monotonic()
        return {'status': 'bot_active'}

    @app.post('/api/chat')
    async def chat(body: ChatInput, request: Request):
        text = body.text.strip()
        if not text:
            raise HTTPException(422, '请输入消息')
        if dataset == 'emanual' and settings.mode == 'fake' and (body.locale != 'en' or any('\u3400' <= c <= '\u9fff' for c in text)):
            raise HTTPException(422, '当前公开说明书为英文，请用英文提问；尚未启用跨语言检索')
        session = sessions.get(request.cookies.get('chat_session', ''))
        if session is None or time.monotonic() - session.touched > 3600:
            raise HTTPException(401, '会话已过期，请新建会话')
        if session.lock.locked():
            raise HTTPException(409, '请等待当前回复完成')
        if session.turns >= 100:
            raise HTTPException(429, '已达到测试会话上限，请新建会话')
        async with session.lock:
            session.turns += 1
            session.touched = time.monotonic()
            validator = DiagnosticReplyValidator()
            memory = await RequestMemorySnapshot.load(
                AsyncInMemoryConversationMemory(session.memory), session.conversation_id,
                limit=settings.summary_retention_turns)
            service = RagAnswerService(
                embedding=app.state.embedding, knowledge=app.state.knowledge, model=app.state.model,
                memory=memory,
                prompt_version=settings.prompt_version, retrieval_top_k=settings.final_candidate_count,
                evidence_threshold=settings.evidence_threshold,
                max_retrieved_chunks=settings.final_candidate_count, max_reply_chars=settings.reply_max_chars,
                recent_turn_limit=settings.summary_retention_turns, clock=lambda: datetime.now(UTC),
                reply_validator=validator)
            dialogue = None
            if app.state.planner is not None:
                dialogue = CustomerDialogue(
                    planner=app.state.planner, memory=memory,
                    rag=service, model=app.state.model, validator=validator,
                    knowledge=app.state.knowledge, search=app.state.search,
                    general_manual_ids=app.state.general_manual_ids)
            worker = ConversationWorker(
                session.conversations, session.outbox, tools=None, identity=None,
                knowledge=None, model=None, templates=None, handoff=FakeHandoff(),
                transaction=InMemoryConversationOutboxTransaction(
                    session.conversations, session.outbox, session.memory),
                rag_answer=dialogue or service, locale=body.locale,
                router=ManualTestRouter() if dialogue is not None or dataset == 'emanual' else None)
            event_id, now = uuid4().hex, datetime.now(UTC)
            # Local ingress is already serialized by the session lock. Coarse
            # Windows clocks can give distinct requests identical timestamps;
            # reserve strictly increasing event time without changing provider
            # duplicate/stale-event rules in the shared worker.
            if session.last_event_at is not None:
                now = max(now, session.last_event_at + timedelta(microseconds=1))
            session.last_event_at = now
            event = InboundEvent(
                event_key=event_id, phone_number_id='web-test', conversation_id=session.conversation_id,
                contact_id=session.conversation_id, recipient_id=session.conversation_id,
                provider_message_id=event_id, event_type=EventType.MESSAGE,
                occurred_at=now, received_at=now, payload_digest=event_id, text=text)
            try:
                result = await worker.process(event)
            except Exception:
                # Never expose provider bodies, credentials, or stack traces to the browser.
                raise HTTPException(503, '处理失败，请新建会话后重试；不会自动重发') from None
            records = await session.outbox.records()
            replies = [r.command.text for r in records if r.command.trace_id == result.trace_id]
            return {'status': result.status, 'reply': replies[0] if replies else None,
                    'reason': (validator.diagnostic['code'] if validator.diagnostic else result.reason),
                    'diagnostic': validator.diagnostic, 'trace_id': result.trace_id,
                    'stage': dialogue.stage if dialogue else None,
                    'sources': dialogue.sources if dialogue else [],
                    'model': result.model_name, 'evidence_count': len(result.supported_chunk_ids)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use configured paid embedding and reply APIs')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--dataset', choices=['sample', 'emanual'], default='sample')
    parser.add_argument('--prepared-root', type=Path, default=Path('.rag-data/prepared-public-v3'))
    args = parser.parse_args()
    from dataclasses import replace
    import uvicorn
    settings = (RagSettings.from_provider_files('.rag-data/bailian.local.json', '.rag-data/deepseek.local.json')
                if args.live else RagSettings())
    knowledge_dir = prepared_manual_directory(args.prepared_root) if args.dataset == 'emanual' else Path('knowledge')
    settings = replace(settings, index_path=f'.rag-index/web-chat-{args.dataset}-{settings.mode}.sqlite3')
    print('Local test only. Live mode uses paid APIs; no WhatsApp/WeChat messages are sent.', flush=True)
    uvicorn.run(create_app(settings, knowledge_dir=str(knowledge_dir), dataset=args.dataset), host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
