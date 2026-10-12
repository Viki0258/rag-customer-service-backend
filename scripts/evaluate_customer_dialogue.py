"""Opt-in live conversational smoke test, with isolated database and sessions.

Uses existing provider configuration; bills intake/embedding/answer API calls.
No channel delivery, production conversation changes or source corpus changes.
Printed replies require human review; evidence counts are not correctness scores.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile

from fastapi.testclient import TestClient

from whatsapp_support.config import RagSettings
from whatsapp_support.cross_language import text_language
from whatsapp_support.web_chat import create_app, prepared_manual_directory


SCENARIOS = {
    'unknown_model': ['三星电视花屏了，怎么办？', '我不知道具体型号，能先帮我排查吗？'],
    'changed_fault': ['三星Q60B画面花屏了', '画面测试正常，现在是没有声音了', '声音测试也没有声音'],
    'fault': ['我昨天买的电视出故障了，该怎么办？', '三星Q60B', '画面花屏了', '具体怎么打开你说的画面测试？', '测试画面是正常的，那接下来呢？'],
    'operation': ['三星电视怎么开启灰度模式？', 'How do I turn it off?'],
    'unknown': ['三星Q60B的维修工程菜单密码是什么？'],
    'danger': ['三星电视冒烟了，我能拆开修吗？'],
    'refund': ['我要退款，请给我办理'],
}


def run(selected):
    settings = RagSettings.from_provider_files('.rag-data/bailian.local.json', '.rag-data/deepseek.local.json')
    source_path = Path('.rag-index/web-chat-emanual-compatible.sqlite3').resolve()
    if not source_path.is_file():
        raise ValueError('Existing live index is required')
    with tempfile.TemporaryDirectory(prefix='rag-dialogue-eval-') as temporary:
        destination = Path(temporary) / 'evaluation.sqlite3'
        source = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        app = create_app(replace(settings, index_path=str(destination)),
                         knowledge_dir=str(prepared_manual_directory(Path('.rag-data/prepared-public-v3'))),
                         dataset='emanual')
        headers = {'X-Chat-Client': '1'}
        with TestClient(app, base_url='http://127.0.0.1') as client:
            for name in selected:
                client.post('/api/session', headers=headers, json={}).raise_for_status()
                locale = 'zh-CN'
                for question in SCENARIOS[name]:
                    locale = text_language(question, locale)
                    response = client.post('/api/chat', headers=headers,
                        json={'text': question, 'locale': locale})
                    if response.status_code != 200:
                        print(json.dumps({'scenario': name, 'http_status': response.status_code}), flush=True)
                        break
                    result = response.json()
                    print(json.dumps({'scenario': name, 'question': question,
                        **{key: result.get(key) for key in ('status', 'reply', 'reason', 'stage', 'evidence_count')}},
                        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--scenario', choices=list(SCENARIOS), action='append')
    args = parser.parse_args()
    run(args.scenario or list(SCENARIOS))
