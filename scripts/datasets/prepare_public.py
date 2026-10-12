"""Prepare downloaded public sources into separate corpora and evaluation splits.

No network, model calls or source modification. Output must be a new directory.
Run from repository root: py -3.12 -m scripts.datasets.prepare_public
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from scripts.datasets import convert_public as convert
from scripts.datasets.download_public import DATASETS, discover_repository_root, resolve_destination
from scripts.eval.run_eval import validate_case


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def save_cases(path, rows):
    for row in rows:
        validate_case(row)
    convert.write_jsonl(rows, path)
    return {'cases': len(rows), 'sha256': digest(path), 'bytes': path.stat().st_size}


def export_documents(directory, documents):
    """Content-addressed exact-text deduplication; maintain every source reference."""
    directory.mkdir(parents=True)
    by_hash = {}
    count = 0
    for source_id, text in documents:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'empty document: {source_id}')
        count += 1
        text = text.strip()
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if key not in by_hash:
            filename = key + '.en.txt'
            (directory / filename).write_text(text, encoding='utf-8')
            by_hash[key] = {'file': filename, 'source_ids': [], 'characters': len(text)}
        by_hash[key]['source_ids'].append(source_id)
    manifest = {'source_documents': count, 'unique_documents': len(by_hash),
                'duplicates_removed': count - len(by_hash), 'documents': by_hash}
    save_json(directory.parent / 'documents.manifest.json', manifest)
    return {key: manifest[key] for key in ('source_documents', 'unique_documents', 'duplicates_removed')}


def deduplicate_bitext(records):
    unique = {}
    for index, row in enumerate(records):
        question = convert._required_text(row, 'instruction', row_number=index)
        intent = convert._required_text(row, 'intent', row_number=index)
        unique.setdefault((intent, question), row)
    labels = defaultdict(set)
    for intent, question in unique:
        labels[question].add(intent)
    return list(unique.values()), len(records) - len(unique), sum(len(v) > 1 for v in labels.values())


def validated_ragbench(rows):
    cases, rejected = [], []
    for row in rows:
        try:
            cases.extend(convert.convert_records('ragbench-emanual', [row]))
        except ValueError as error:
            rejected.append({'id': row.get('id'), 'reason': str(error),
                             'all_relevant_sentence_keys': row.get('all_relevant_sentence_keys')})
    ids = [case['case_id'] for case in cases]
    if len(ids) != len(set(ids)) or not cases:
        raise ValueError('duplicate or empty RAGBench cases')
    return sorted(cases, key=lambda case: case['case_id']), rejected


def prepare(source_dir: Path, output: Path):
    source_files = {'multidoc2dial': 'multidoc2dial.zip',
                    'ragbench-emanual': 'emanual-test.parquet', 'bitext-customer-support': 'bitext.csv'}
    provenance = {}
    # Verify source identities and exact bytes before publishing any derived data.
    for name, filename in source_files.items():
        path = source_dir / filename
        metadata = json.loads(path.with_name(path.name + '.metadata.json').read_text(encoding='utf-8'))
        from scripts.datasets.download_public import _authoritative_identity, _matching_existing_download
        if not _matching_existing_download(
            path, path.with_name(path.name + '.metadata.json'),
            expected_identity=_authoritative_identity(name, DATASETS[name]),
        ):
            raise FileNotFoundError(path)
        provenance[name] = metadata
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'sources': provenance, 'datasets': {},
              'scope': 'Data preparation only; no embedding or generated-answer evaluation.',
              'locale': 'en', 'answers_in_knowledge': False}
    save_json(output / 'report.json', report)

    # Full source documents, not answers or only gold-positive snippets.
    path = source_dir / 'multidoc2dial.zip'
    with zipfile.ZipFile(path) as archive:
        members = convert._validate_zip_members(archive)
        doc_member = next(m for m in members if PurePosixPath(m.filename).name == 'multidoc2dial_doc.json')
        docs = convert._read_zip_json(archive, doc_member)
        documents = []
        domains = Counter()
        for domain, mapping in sorted(docs['doc_data'].items()):
            for doc_id, doc in sorted(mapping.items()):
                documents.append((f'{domain}:{doc_id}', doc['doc_text']))
                domains[domain] += 1
        entry = export_documents(output / 'multidoc2dial' / 'knowledge', documents)
        entry['domains'] = dict(domains)
        entry['splits'] = {}
        for member in sorted(members, key=lambda m: m.filename):
            name = PurePosixPath(member.filename).name
            if not name.startswith('multidoc2dial_dial_') or not name.endswith('.json'):
                continue
            split = name.removeprefix('multidoc2dial_dial_').removesuffix('.json')
            payload = convert._read_zip_json(archive, member)
            records = convert._multidoc_records(docs, [(member.filename, payload)])
            cases = convert.convert_records('multidoc2dial', records)
            entry['splits'][split] = save_cases(output / 'multidoc2dial' / f'{split}.cases.jsonl', cases)
            # Sidecar restores dialogue ordering/context absent from the flat case schema.
            contexts = []
            for domain, dialogues in payload['dial_data'].items():
                for dialogue in dialogues:
                    history = []
                    for turn in sorted(dialogue['turns'], key=lambda t: int(t['turn_id'])):
                        if turn['role'] == 'agent' and turn.get('references'):
                            contexts.append({'case_id': f"multidoc2dial:{dialogue['dial_id']}:{turn['turn_id']}",
                                             'split': split, 'domain': domain, 'dialogue_id': dialogue['dial_id'],
                                             'preceding_turns': list(history)})
                        history.append({'role': turn['role'], 'text': turn['utterance']})
            convert.write_jsonl(contexts, output / 'multidoc2dial' / f'{split}.context.jsonl')
            del cases, records
        report['datasets']['multidoc2dial'] = entry

    rows = convert.read_records(source_dir / 'emanual-test.parquet')
    cases, rejected = validated_ragbench(rows)
    entry = export_documents(output / 'ragbench-emanual' / 'knowledge', (
        (f"{row['id']}:{row.get('generation_model_name', '')}:document:{index}", text)
        for row in rows for index, text in enumerate(row['documents'])))
    entry['test'] = save_cases(output / 'ragbench-emanual' / 'test.cases.jsonl', cases)
    entry['raw_cases'] = len(rows)
    entry['unique_source_question_ids'] = len({row['id'] for row in rows})
    entry['quarantined_cases'] = rejected
    save_json(output / 'ragbench-emanual' / 'quarantine.json', rejected)
    entry['sample_questions'] = [case['question'] for case in cases[:8]]
    entry['note'] = 'Pooled candidate documents from the test split, not a complete product catalog. No source response is indexed.'
    report['datasets']['ragbench-emanual'] = entry

    records = convert.read_records(source_dir / 'bitext.csv')
    unique, duplicates, conflicts = deduplicate_bitext(records)
    cases = convert.convert_records('bitext-customer-support', unique)
    entry = save_cases(output / 'bitext-customer-support' / 'train.cases.jsonl', cases)
    entry.update({'raw_rows': len(records), 'duplicate_intent_instruction_pairs_removed': duplicates,
                  'instructions_with_multiple_intents': conflicts,
                  'intent_counts': dict(Counter(case['expected_intent'] for case in cases)),
                  'knowledge_documents': 0, 'note': 'Routing-only; source responses intentionally excluded.'})
    report['datasets']['bitext-customer-support'] = entry
    report['status'] = 'complete'
    save_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('.rag-data/public'))
    parser.add_argument('--output', type=Path, default=Path('.rag-data/prepared-public-v3'))
    args = parser.parse_args()
    root = discover_repository_root()
    output = resolve_destination(args.output, root)
    report = prepare(args.source_dir.resolve(), output)
    print(json.dumps({'status': report['status'], 'datasets': report['datasets'], 'report': str(output / 'report.json')}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
