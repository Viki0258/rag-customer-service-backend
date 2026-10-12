"""Conservative policy follow-ups, using customer history only as search context."""
from collections.abc import Sequence

from whatsapp_support.domain import ConversationTurn
from whatsapp_support.policy import IntentRouter


def is_policy_followup(text: str) -> bool:
    normalized = ' '.join(text.casefold().strip().rstrip('?.!？。！').split())
    if normalized.startswith('and '):
        normalized = normalized[4:]
    return normalized in {
        'and how many days do i have', 'how many days do i have',
        'and how long does it take', 'how long does it take',
        'what about the conditions', 'what are the conditions',
        '那要几天', '要几天', '那多久', '需要多久', '那有什么条件', '有什么条件',
    }


def policy_followup_query(question: str, turns: Sequence[ConversationTurn]) -> str | None:
    if not is_policy_followup(question):
        return None
    router = IntentRouter()
    for turn in reversed(turns):
        previous = turn.customer_text.strip()
        if is_policy_followup(previous):
            continue
        # Stop at a topic change; never mine older policy topics past other requests.
        if len(previous) > 500 or router.route(previous) not in {'product_info', 'product_policy'}:
            return None
        return f'{previous}\nFollow-up: {question}'
    return None
