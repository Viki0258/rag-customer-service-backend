from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from whatsapp_support.customer_dialogue import CustomerDialogue, Intake
from whatsapp_support.manual_scope import GeneralManualKnowledge
from whatsapp_support.domain import RetrievalQuery


@pytest.mark.asyncio
@pytest.mark.parametrize('brand,symptom,ids,expected', [
    ('Samsung', 'distorted picture', ['reviewed'], True),
    ('TCL', 'distorted picture', ['reviewed'], False),
    ('', 'distorted picture', ['reviewed'], False),
    ('Samsung', '', ['reviewed'], False),
    ('Samsung', 'distorted picture', [], False),
])
async def test_unknown_model_can_use_only_confirmed_brand_and_reviewed_general_evidence(brand, symptom, ids, expected):
    from dataclasses import make_dataclass
    from whatsapp_support.customer_dialogue import ZH_QUESTIONS
    general = SimpleNamespace(answer=AsyncMock(return_value='general-answer'))
    rag = SimpleNamespace(with_knowledge=lambda *args, **kwargs: general)
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=[SimpleNamespace(
        customer_text='电视花屏了', assistant_text=ZH_QUESTIONS['model'])]),
        get_summary=AsyncMock(return_value=None))
    service = CustomerDialogue(planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(
        topic='fault', brand=brand, symptom=symptom))), memory=memory, rag=rag,
        model=None, validator=None, knowledge=None, general_manual_ids=ids)
    event = make_dataclass('Event', [('text', str), ('conversation_id', str)])('不知道型号，能先排查吗？', 'c')
    result = await service.answer(event, 'zh-CN', 'trace')
    assert general.answer.call_count == int(expected)
    if expected:
        assert result == 'general-answer'
        assert event.text in general.answer.call_args.args[0].text
    else:
        assert result.reason == 'clarification_exhausted'


def test_general_manual_prompt_requires_progressive_grounded_troubleshooting():
    from datetime import UTC, datetime
    from whatsapp_support.manual_scope import GeneralManualPrompt
    from whatsapp_support.memory import ConversationContext
    context = ConversationContext(summary=None, recent_turns=(), handoff_state='none',
                                  latest_customer_message='My TV picture is distorted')
    request = GeneralManualPrompt().build(context, context.latest_customer_message, [],
        'en', 'test', 'conversation', 'trace', datetime.now(UTC))
    system = request.inputs['messages'][0]['content']
    assert 'single most relevant supported user-level diagnostic check' in system
    assert 'customer confirms the condition' in system
    assert 'without adding fault troubleshooting or an unnecessary question' in system
    assert 'NOT confirmed documentation for the named model' in system
    assert 'Do not add steps from model memory' in system
    assert 'warm, concise customer-service voice' in system
    assert 'without promising that matching documentation is available' in system
    assert 'Do not request a model already supplied' in system
    assert 'ask at most one useful question per reply' in system
    assert 'Do not claim verified model compatibility or hide uncertainty' in system
    assert 'Explicitly say that exact-model applicability is unverified' not in system


@pytest.mark.asyncio
async def test_scoped_retrieval_finds_evidence_below_global_top_five(tmp_path):
    """Ranking unrelated documents first must not erase approved evidence."""
    from whatsapp_support.rag_index import SqliteKnowledgeIndex, SqliteHybridKnowledge, KnowledgeFileLoader

    class Embedding:
        model_name = 'test-scope'
        dimension = 2

        async def embed(self, texts):
            return [[1.0, 0.0] if 'unreviewed' in text else [0.5, 0.5] for text in texts]

    for number in range(25):
        (tmp_path / f'noise-{number}.en.txt').write_text(
            'unreviewed picture test screen troubleshooting', encoding='utf-8')
    (tmp_path / 'approved.en.txt').write_text('Reviewed manual: open Support then Self Diagnosis.', encoding='utf-8')
    document_id = next(d.document_id for d in KnowledgeFileLoader.load_directory(tmp_path)
                       if d.title == 'approved')
    index = SqliteKnowledgeIndex(tmp_path / 'index.sqlite3')
    await index.open()
    try:
        await index.import_directory(tmp_path, Embedding())
        base = SqliteHybridKnowledge(index, final_k=5)
        query = RetrievalQuery('picture test', 'c', 5, locale='en',
                               query_vector=(1.0, 0.0), embedding_model='test-scope', embedding_dimension=2)
        assert all(c.document_id != document_id for c in await base.retrieve(query))
        scoped = GeneralManualKnowledge(base, [document_id])
        hits = await scoped.retrieve(query)
        assert [c.document_id for c in hits] == [document_id]
        assert await GeneralManualKnowledge(base, []).retrieve(query) == []
        assert all(c.document_id != document_id for c in await base.retrieve(query))
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_general_evidence_is_explicitly_allowlisted():
    approved = SimpleNamespace(document_id='reviewed', text='Original manual text')
    other = SimpleNamespace(document_id='other-model', text='Unreviewed instructions')
    knowledge = SimpleNamespace(retrieve=AsyncMock(return_value=[other, approved]))
    scoped = GeneralManualKnowledge(knowledge, ['reviewed'])
    assert await scoped.retrieve(RetrievalQuery('picture test', 'c', 5)) == [approved]


@pytest.mark.asyncio
async def test_scope_intersects_existing_restrictions_and_survives_translation():
    from whatsapp_support.cross_language import CrossLanguageKnowledge
    approved = SimpleNamespace(document_id='reviewed', chunk_id='one', fused_score=0.1)
    base = SimpleNamespace(retrieve=AsyncMock(return_value=[approved]))
    embedding = SimpleNamespace(embed=AsyncMock(return_value=[[1.0, 0.0]]))
    translator = SimpleNamespace(translate=AsyncMock(return_value='picture test'))
    cross = CrossLanguageKnowledge(base, embedding, translator, ['en'])
    scoped = GeneralManualKnowledge(cross, ['reviewed', 'another'])
    query = RetrievalQuery('画面自检', 'c', 5, locale='zh-CN', allowed_document_ids=('reviewed', 'outside'))
    assert await scoped.retrieve(query) == [approved]
    assert base.retrieve.call_args.args[0].allowed_document_ids == ('reviewed',)
    base.retrieve.reset_mock()
    from dataclasses import replace
    assert await scoped.retrieve(replace(query, allowed_document_ids=())) == []
    base.retrieve.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('model,reason,fallback', [
    ('Samsung Q60B', 'insufficient_evidence', True),
    ('Samsung 75Q7F', 'insufficient_evidence', True),
    ('TCL 85T7H', 'insufficient_evidence', False),
    ('Samsung Q60B', 'model_failed', False),
    ('Samsung Q60B', 'reply_validation_failed', False),
])
async def test_fault_fallback_only_for_samsung_missing_model_evidence(model, reason, fallback):
    exact = SimpleNamespace(answer=AsyncMock(return_value=SimpleNamespace(reason=reason)))
    general = SimpleNamespace(answer=AsyncMock(return_value='general-answer'))
    rag = SimpleNamespace(with_knowledge=lambda knowledge, **kwargs: general if kwargs else exact)
    service = CustomerDialogue(
        planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='fault', model=model, symptom='distorted picture'))),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None)),
        rag=rag, model=None, validator=None, knowledge=None, general_manual_ids=['reviewed'])
    # Production inbound events are dataclasses because the dialogue preserves them with replace.
    from dataclasses import make_dataclass
    event = make_dataclass('Event', [('text', str), ('conversation_id', str)])('TV has distorted picture', 'c')
    result = await service.answer(event, 'en', 'trace')
    assert general.answer.call_count == int(fallback)
    if fallback:
        assert result == 'general-answer'
        assert 'unverified' in general.answer.call_args.args[0].text


@pytest.mark.asyncio
async def test_operation_goes_to_local_rag_not_web_search():
    from dataclasses import make_dataclass
    general = SimpleNamespace(answer=AsyncMock(return_value='answer'))
    rag = SimpleNamespace(with_knowledge=lambda *a, **kw: general)
    search = SimpleNamespace(search=AsyncMock())
    service = CustomerDialogue(
        planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='knowledge', model='Samsung Q60B',
            search_goal='operation', resolved_question='How to run Sound Test?'))),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None)),
        rag=rag, model=None, validator=None, knowledge=None, search=search, general_manual_ids=['reviewed'])
    event = make_dataclass('Event', [('text', str), ('conversation_id', str)])('声音自检怎么开', 'c')
    assert await service.answer(event, 'zh-CN', 'trace') == 'answer'
    search.search.assert_not_called()


@pytest.mark.asyncio
async def test_fault_followup_preserves_current_test_result_in_both_rag_paths():
    from dataclasses import make_dataclass
    exact = SimpleNamespace(answer=AsyncMock(return_value=SimpleNamespace(reason='insufficient_evidence')))
    general = SimpleNamespace(answer=AsyncMock(return_value='answer'))
    rag = SimpleNamespace(with_knowledge=lambda knowledge, **kwargs: general if kwargs else exact)
    service = CustomerDialogue(
        planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='fault', model='Samsung Q60B',
            symptom='distorted picture', resolved_question='The picture test is normal; what next?'))),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None)),
        rag=rag, model=None, validator=None, knowledge=None, general_manual_ids=['reviewed'])
    event = make_dataclass('Event', [('text', str), ('conversation_id', str)])('测试画面正常，接下来呢？', 'c')
    assert await service.answer(event, 'zh-CN', 'trace') == 'answer'
    for path in (exact, general):
        forwarded = path.answer.call_args.args[0].text
        assert event.text in forwarded
        assert 'The picture test is normal; what next?' in forwarded
    assert event.text == '测试画面正常，接下来呢？'


@pytest.mark.asyncio
@pytest.mark.parametrize('model,brand,expected', [('Q60B', 'Samsung', True),
    ('Samsung Q60B', 'TCL', False), ('Q60B', '', False)])
async def test_fault_brand_is_separate_from_model_code(model, brand, expected):
    from dataclasses import make_dataclass
    exact = SimpleNamespace(answer=AsyncMock(return_value=SimpleNamespace(reason='insufficient_evidence')))
    general = SimpleNamespace(answer=AsyncMock(return_value='general-answer'))
    service = CustomerDialogue(
        planner=SimpleNamespace(plan=AsyncMock(return_value=Intake(topic='fault', model=model,
            brand=brand, symptom='distorted picture'))),
        memory=SimpleNamespace(get_recent_turns=AsyncMock(return_value=[]), get_summary=AsyncMock(return_value=None)),
        rag=SimpleNamespace(with_knowledge=lambda knowledge, **kwargs: general if kwargs else exact),
        model=None, validator=None, knowledge=None, general_manual_ids=['reviewed'])
    event = make_dataclass('Event', [('text', str), ('conversation_id', str)])('画面花屏了', 'c')
    await service.answer(event, 'zh-CN', 'trace')
    assert general.answer.call_count == int(expected)
