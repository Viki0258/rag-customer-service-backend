"""One bounded return-time clarification; no model-generated business facts."""
from collections.abc import Sequence

from whatsapp_support.domain import ConversationTurn
from whatsapp_support.followups import policy_followup_query
from whatsapp_support.policy import IntentRouter

QUESTIONS = {
    'zh': '您是问退货申请期限，还是办理退货需要多久？',
    'en': 'Do you mean the deadline to request a return, or how long processing a return takes?',
}


def normalized(text: str) -> str:
    return ' '.join(text.casefold().strip().rstrip('?.!？。！').split())


def pending_clarification(turns: Sequence[ConversationTurn]) -> bool:
    # Only trusted assistant history counts, not customer messages or summary text.
    return bool(turns and turns[-1].assistant_text.strip() in QUESTIONS.values())


def clarification_choice(question: str, turns: Sequence[ConversationTurn]) -> str | None:
    if not pending_clarification(turns):
        return None
    choice = normalized(question)
    if choice in {'申请期限', '退货申请期限', '第一个', '前者', 'deadline',
                  'the deadline', 'return deadline', 'the first one'}:
        return 'deadline'
    if choice in {'办理时长', '办理退货需要多久', '第二个', '后者',
                  'processing time', 'the second one'}:
        return 'processing'
    # Explicit new topics still take the normal route. Unknown answers do not
    # trigger another clarification; they exhaust this clarification episode.
    if IntentRouter().route(question) is not None:
        return None
    return 'unresolved'


def needs_return_time_clarification(question: str, turns: Sequence[ConversationTurn]) -> bool:
    if pending_clarification(turns):
        return False
    if normalized(question) not in {'那要几天', '要几天', '那多久', '需要多久',
                                     'and how long does it take', 'how long does it take'}:
        return False
    query = policy_followup_query(question, turns)
    if query is None:
        return False
    topic = query.split('\nFollow-up:', 1)[0].casefold()
    return 'return' in topic or '退货' in topic
