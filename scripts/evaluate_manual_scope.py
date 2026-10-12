"""Compare old post-top-k filtering with scoped retrieval on the existing index.

Run from repository root with PYTHONPATH=src. Requires --live acknowledgement;
embeds six public test questions (billable), does not generate answers or write data.
This is a small retrieval regression set, not an answer-accuracy benchmark.
"""
import argparse
import asyncio
from dataclasses import replace
import json

from whatsapp_support.config import RagSettings
from whatsapp_support.domain import RetrievalQuery
from whatsapp_support.manual_scope import GENERAL_MANUAL_CHECKSUMS, GeneralManualKnowledge
from whatsapp_support.rag_index import SqliteKnowledgeIndex, SqliteHybridKnowledge


CASES = [
    ('How do I enable grayscale on my Samsung TV?', '0f4a2c'),
    ('How do I start a picture test on my Samsung TV?', '588489'),
    ('How do I run a sound test on my Samsung TV?', '422d08'),
    ('Where can I find the model and software version of my Samsung TV?', '6d5820'),
    ('Samsung TV symptom: distorted picture. What general user-level checks do the supplied manual excerpts describe?', '588489'),
    ('Samsung TV symptom: no sound. What general user-level checks do the supplied manual excerpts describe?', '422d08'),
]


async def evaluate():
    settings = RagSettings.from_provider_files('.rag-data/bailian.local.json', '.rag-data/deepseek.local.json')
    index = SqliteKnowledgeIndex('.rag-index/web-chat-emanual-compatible.sqlite3')
    embedding = settings.create_embedding()
    try:
        await index.open_readonly(live_updates=True)
        snapshot = await index.inspect_active()
        generation = snapshot['generation']
        if not generation or generation['embedding_model'] != embedding.model_name or generation['embedding_dimension'] != embedding.dimension:
            raise ValueError('Index/model mismatch; no API request made')
        docs = snapshot['documents']
        allowed = {d['document_id'] for d in docs if d['checksum'] in GENERAL_MANUAL_CHECKSUMS}
        expected = [{d['document_id'] for d in docs if d['checksum'].startswith(prefix)} for _, prefix in CASES]
        if not all(expected):
            raise ValueError('Required evaluation source documents are missing')
        base = SqliteHybridKnowledge(index, vector_candidate_k=settings.vector_candidate_count,
            bm25_candidate_k=settings.bm25_candidate_count, final_k=settings.final_candidate_count,
            threshold=settings.evidence_threshold)
        scoped = GeneralManualKnowledge(base, allowed)
        vectors = await embedding.embed([question for question, _ in CASES])
        if len(vectors) != len(CASES):
            raise ValueError('Unexpected embedding count')
        results = []
        for (question, _), vector, targets in zip(CASES, vectors, expected):
            query = RetrievalQuery(question, 'scope-evaluation', settings.final_candidate_count,
                locale='en', query_vector=tuple(vector), embedding_model=embedding.model_name,
                embedding_dimension=embedding.dimension)
            before = [c for c in await base.retrieve(replace(query, top_k=max(20, query.top_k)))
                      if c.document_id in allowed][:query.top_k]
            after = await scoped.retrieve(query)
            def report(chunks):
                return {'expected_document_hit': any(c.document_id in targets for c in chunks),
                        'chunks': len(chunks), 'document_ids': list(dict.fromkeys(c.document_id for c in chunks))}
            results.append({'question': question, 'before': report(before), 'after': report(after)})
        active = await index.active_generation()
        if active is None or active.generation_id != generation['generation_id']:
            raise ValueError('Index changed during evaluation')
        print(json.dumps({'generation': generation['generation_id'], 'cases': results,
            'before_hits': sum(r['before']['expected_document_hit'] for r in results),
            'after_hits': sum(r['after']['expected_document_hit'] for r in results),
            'total': len(results)}, ensure_ascii=False, indent=2))
    finally:
        await embedding.aclose()
        await index.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.parse_args()
    asyncio.run(evaluate())
