"""Content-free, per-request validation diagnostics for the local playground only."""
from whatsapp_support.rag import (
    RagReplyValidator, _is_valid_model_payload, _normalize_reply_text,
    _reply_matches_locale, _contains_internal_leakage,
)
from whatsapp_support.rag_retrieval import normalize_locale


class DiagnosticReplyValidator(RagReplyValidator):
    def __init__(self):
        self.diagnostic = None

    def validate(self, model_result, retrieved_chunks, locale, trace_id, limits):
        result = super().validate(model_result, retrieved_chunks, locale, trace_id, limits)
        payload = model_result.content
        valid = _is_valid_model_payload(payload)
        diagnostic = {'schema_valid': valid, 'code': 'model_schema_invalid',
                      'max_reply_chars': limits.max_reply_chars}
        if valid:
            reply = _normalize_reply_text(payload['reply'])
            retrieved_ids = tuple(chunk.chunk_id for chunk in retrieved_chunks)
            ids = payload['supported_evidence_ids']
            membership = all(value in retrieved_ids for value in ids)
            checks = {
                'reply_empty': not reply,
                'reply_too_long': len(reply) > limits.max_reply_chars,
                'reply_locale_mismatch': not _reply_matches_locale(reply, normalize_locale(locale)),
                'reply_internal_leakage': _contains_internal_leakage(
                    reply, (*retrieved_ids, trace_id, *limits.internal_values)),
            }
            # Same decision order as the production validator. Do not expose the
            # model's free-form reason, candidate text, prompts, IDs, or secrets.
            if not membership:
                code = 'unsupported_evidence'
            elif payload['handoff']:
                code = 'model_requested_handoff'
            elif not ids:
                code = 'missing_evidence'
            else:
                code = next((key for key, failed in checks.items() if failed), 'validated')
            diagnostic.update(code=code, reply_chars=len(reply),
                              model_requested_handoff=payload['handoff'],
                              cited_count=len(ids), retrieved_count=len(retrieved_ids),
                              citation_membership_valid=membership,
                              failed_reply_checks=[key for key, failed in checks.items() if failed])
        self.diagnostic = diagnostic
        return result
