import pytest

from whatsapp_support.policy import IntentRouter


@pytest.mark.parametrize('text', [
    'And what are the conditions for a return?',
    'What are the requirements for returning an item?',
    'What conditions apply to returns?',
    'Tell me the return requirements.',
    'What is the deadline for returns?',
    '那退货需要满足什么条件？',
])
def test_general_return_questions_are_public_knowledge(text):
    assert IntentRouter().route(text) == 'product_policy'


@pytest.mark.parametrize('text,intent', [
    ('What are the conditions for a return? Refund me now.', 'refund'),
    ('What are the conditions for a return? Can I return this?', 'return_eligibility'),
    ('Tell me the return requirements and cancel my order.', 'cancel_order'),
    ('那退货需要什么条件？请退款。', 'refund'),
    ('Can I return this item?', 'return_eligibility'),
    ('What are the conditions for a loan?', None),
    ('return x + y', None),
])
def test_public_question_patterns_do_not_override_protection(text, intent):
    assert IntentRouter().route(text) == intent


def test_explicit_return_topic_is_not_a_vague_shipping_followup():
    from types import SimpleNamespace
    from whatsapp_support.followups import policy_followup_query
    turns = [SimpleNamespace(customer_text='What is the return policy?'),
             SimpleNamespace(customer_text='And what are the conditions for a return?')]
    query = policy_followup_query('And what are the conditions?', turns)
    assert 'conditions for a return' in query
    assert policy_followup_query('And what are the conditions?', []) is None
    assert policy_followup_query('And what are the conditions for a return?',
        [SimpleNamespace(customer_text='Standard shipping time?')]) is None
