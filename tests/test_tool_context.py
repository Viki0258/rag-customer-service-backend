from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from whatsapp_support.orchestration import resolve_tool_reference, ConversationOrchestrator
from whatsapp_support.policy import IntentRouter, SafetyPolicy
from whatsapp_support.replies import ReplyComposer
from test_orchestration import make_event, make_conversation, NOW


def turn(customer, assistant=''):
    return SimpleNamespace(customer_text=customer, assistant_text=assistant)


def test_reference_follows_customer_not_assistant_and_current_override_wins():
    history = [turn('Is sku-1 in stock?', 'sku-999 is available')]
    assert resolve_tool_reference('inventory', 'Is it in stock?', history).endswith('sku-1')
    assert resolve_tool_reference('inventory', 'Is sku-2 in stock?', history) == 'Is sku-2 in stock?'


@pytest.mark.parametrize('history', [[], [turn('hello', 'sku-1')],
    [turn('sku-1 and sku-2')], [turn('sku-1'), turn('Please refund my order')]])
def test_no_reference_guessing(history):
    with pytest.raises(ValueError):
        resolve_tool_reference('inventory', 'Is it in stock?', history)


@pytest.mark.asyncio
async def test_history_does_not_authorize_order_query():
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=[turn('order-1')]))
    tool = SimpleNamespace(execute=AsyncMock())
    service = ConversationOrchestrator(tools=tool, identity=None, router=IntentRouter(),
        policy=SafetyPolicy(), composer=ReplyComposer(), memory=memory, clock=lambda: NOW)
    result = await service.run(make_event('What is the status of that order?'), make_conversation())
    assert result.status == 'handoff'
    tool.execute.assert_not_called()
    memory.get_recent_turns.assert_not_called()


@pytest.mark.asyncio
async def test_inventory_uses_context_but_rejects_mismatched_result():
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=[turn('sku-1')]))
    tool = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        success=True, output_data={'product_id': 'sku-2', 'available': True})))
    service = ConversationOrchestrator(tools=tool, identity=None, router=IntentRouter(),
        policy=SafetyPolicy(), composer=ReplyComposer(), memory=memory, clock=lambda: NOW)
    result = await service.run(make_event('Is it in stock?'), make_conversation())
    assert tool.execute.call_args.args[0].arguments == {'product_id': 'sku-1'}
    assert result.reason == 'tool_target_mismatch'
    memory.get_recent_turns.assert_awaited_once_with('conversation-1', limit=32)


@pytest.mark.asyncio
async def test_repeat_read_after_seven_turns_resolves_target_and_rechecks_tool():
    history = [turn('Is sku-1 in stock?')] + [turn('thanks')] * 7
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=history))
    tool = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        success=True, output_data={'product_id':'sku-2'})))
    service = ConversationOrchestrator(tools=tool, identity=None, router=IntentRouter(),
        policy=SafetyPolicy(), composer=ReplyComposer(), memory=memory, clock=lambda: NOW)
    result = await service.run(make_event('check again'), make_conversation())
    assert tool.execute.call_args.args[0].arguments['product_id'] == 'sku-1'
    assert result.reason == 'tool_target_mismatch'


@pytest.mark.asyncio
async def test_repeat_protected_read_still_needs_authorization():
    memory = SimpleNamespace(get_recent_turns=AsyncMock(return_value=[turn('order status order-1')]))
    tool = SimpleNamespace(execute=AsyncMock())
    service = ConversationOrchestrator(tools=tool, identity=None, router=IntentRouter(),
        policy=SafetyPolicy(), composer=ReplyComposer(), memory=memory, clock=lambda: NOW)
    result = await service.run(make_event('and now?'), make_conversation())
    assert result.reason == 'identity_required'
    tool.execute.assert_not_called()
