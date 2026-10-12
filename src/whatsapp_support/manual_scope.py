"""Applicability for existing, reviewed eManual excerpts, not invented model mappings."""
from dataclasses import replace

from whatsapp_support.rag import RagPromptBuilder


# SHA256 of the existing public dataset documents. No text is added to the corpus.
GENERAL_MANUAL_CHECKSUMS = frozenset({
    '58848970ed6f7e88119644a24bc8fea5aef994923cbd1f6c45232abd259060ad',  # picture test
    '422d081516bd1cce21cef67e5eb4a42f1a17372894b824be6313dce52eb67b80',  # sound test
    '0f4a2c01c63c7750feca529cf9bbc26870c92eb17582e544e03e9c6f54baa796',  # grayscale
    '6d5820ee6e37bce86f5991976ff9253f3c1030b9dd8361c55cd8f48c0cbeafff',  # model/software info
})


class GeneralManualKnowledge:
    def __init__(self, knowledge, document_ids):
        self.knowledge = knowledge
        self.document_ids = frozenset(document_ids)

    async def retrieve(self, query):
        allowed = self.document_ids
        if query.allowed_document_ids is not None:
            allowed = allowed.intersection(query.allowed_document_ids)
        if not allowed:
            return []
        chunks = await self.knowledge.retrieve(replace(
            query, allowed_document_ids=tuple(sorted(allowed))))
        # Keep a defensive post-check for adapters without pre-filter support.
        return [chunk for chunk in chunks if chunk.document_id in allowed][:query.top_k]


class GeneralManualPrompt(RagPromptBuilder):
    def build(self, *args, **kwargs):
        request = super().build(*args, **kwargs)
        request.inputs['messages'][0]['content'] += (
            ' These are general Samsung manual excerpts, NOT confirmed documentation for the named model. '
            'Answer only general user-accessible operation or diagnostic steps supported by these excerpts. '
            'Use a warm, concise customer-service voice, addressing Chinese customers as 您. '
            'Convey applicability through helpful conditional wording, not a technical disclaimer '
            'such as 确切机型是否适用尚未核实. Do not append a repetitive 请注意 paragraph. '
            'Phrase menu steps conditionally: if the option exists on the TV. '
            'For example: 您可以先看看电视里是否有以下选项： followed by the supported menu path. '
            'When useful, invite the customer to share their model so you can look for relevant '
            'information, without promising that matching documentation is available. For example: '
            '如果找不到这个选项，您可以告诉我具体型号，我再帮您查找对应资料。 '
            'Do not request a model already supplied in the conversation, already requested, or '
            'which the customer said they do not know. Do not add a model request when asking '
            'about the diagnostic result; ask at most one useful question per reply. '
            'Never claim a model supports a feature based on these general excerpts. '
            'Do not provide firmware, service-menu codes, disassembly, electrical repair, or model-specific '
            'hardware instructions. If the question requires those, request handoff. '
            'Do not add steps from model memory. Missing model documentation alone is not a reason '
            'to withhold supported general instructions with the stated limitations. '
            'Use progressive troubleshooting: for an initial fault report, select the single most '
            'relevant supported user-level diagnostic check, not a checklist of every possible cause. '
            'When a picture or sound self-test is relevant and documented, start there with its '
            'documented menu path, then ask at most one question about the test result. '
            'Only give conditional follow-up steps after the customer reports the result or '
            'explicitly asks for alternatives. Never assume they use component cables, a set-top '
            'box or a particular signal source. Keep such branch-specific instructions out until '
            'the customer confirms the condition. For a direct how-to question, answer that action '
            'without adding fault troubleshooting or an unnecessary question. '
            'Do not claim verified model compatibility or hide uncertainty. Preserve direction and bounds '
            'of any numerical condition exactly; omit a detail if it cannot be translated clearly.')
        return request
