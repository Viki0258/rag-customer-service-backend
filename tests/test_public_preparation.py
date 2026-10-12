import json

import pytest

from scripts.datasets.convert_public import convert_records
from scripts.datasets.prepare_public import deduplicate_bitext, export_documents, save_cases, validated_ragbench


def publisher_row():
    return {'id': 'test', 'question': 'What is supported?',
            'documents_sentences': [[['0a', 'The supported feature.'], ['0b', 'Other feature.']]],
            'all_relevant_sentence_keys': ['0a'], 'unsupported_response_sentence_keys': ['b'],
            'sentence_support_information': [
                {'response_sentence_key': 'a', 'supporting_sentence_keys': ['0a'], 'fully_supported': None},
                {'response_sentence_key': 'b', 'supporting_sentence_keys': [], 'fully_supported': False}],
            'response': 'DO NOT INDEX THIS GENERATED ANSWER', 'adherence_score': False}


def test_publisher_parquet_shape_preserves_ids_and_labels(tmp_path):
    row = convert_records('ragbench-emanual', [publisher_row()])[0]
    assert row['evidence_candidates']['test:sentence:0a']['expected_relevant'] is True
    assert row['support_labels'] == {'a': 'supported', 'b': 'unsupported'}
    assert row['source_annotations']['adherence_score'] == 0.0
    assert row['source_annotations']['sentence_support_information'] == {'a': ['0a'], 'b': []}
    assert 'DO NOT INDEX' not in json.dumps(row)
    assert save_cases(tmp_path / 'cases.jsonl', [row])['cases'] == 1


def test_duplicate_publisher_keys_fail():
    row = publisher_row()
    row['documents_sentences'][0].append(['0a', 'Conflicting text'])
    with pytest.raises(ValueError, match='duplicate sentence keys'):
        convert_records('ragbench-emanual', [row])


def test_invalid_source_gold_is_quarantined_not_silently_rewritten():
    invalid = publisher_row()
    invalid['id'] = 'bad'
    invalid['all_relevant_sentence_keys'] = ['0a.']
    cases, rejected = validated_ragbench([publisher_row(), invalid])
    assert len(cases) == 1
    assert rejected[0]['id'] == 'bad'
    assert rejected[0]['all_relevant_sentence_keys'] == ['0a.']
    row = publisher_row()
    row['sentence_support_information'].append(row['sentence_support_information'][0])
    with pytest.raises(ValueError, match='duplicate response sentence key'):
        convert_records('ragbench-emanual', [row])


def test_source_answer_variants_have_distinct_stable_ids():
    first = dict(publisher_row(), generation_model_name='model-a')
    second = dict(publisher_row(), generation_model_name='model-b')
    cases, rejected = validated_ragbench([first, second])
    assert len(cases) == 2 and not rejected
    assert cases == validated_ragbench([second, first])[0]


def test_deduplication_keeps_conflicting_intents_visible():
    records = [{'instruction': 'Track it', 'intent': 'track_order', 'response': 'one'},
               {'instruction': 'Track it', 'intent': 'track_order', 'response': 'two'},
               {'instruction': 'Track it', 'intent': 'other'}]
    unique, removed, conflicts = deduplicate_bitext(records)
    assert len(unique) == 2
    assert removed == 1
    assert conflicts == 1
    assert all(not row['evidence_candidates'] for row in convert_records('bitext-customer-support', unique))


def test_corpus_export_deduplicates_text_not_provenance(tmp_path):
    directory = tmp_path / 'knowledge'
    report = export_documents(directory, [('a:0', 'Example.'), ('b:0', 'Example.'), ('c:0', 'Different.')])
    assert report == {'source_documents': 3, 'unique_documents': 2, 'duplicates_removed': 1}
    manifest = json.loads((tmp_path / 'documents.manifest.json').read_text())
    assert sorted(len(item['source_ids']) for item in manifest['documents'].values()) == [1, 2]
    assert len(list(directory.glob('*.en.txt'))) == 2
