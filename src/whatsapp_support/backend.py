"""Start the shared customer-service HTTP API without any frontend assets."""
import argparse
from dataclasses import replace
import os
from pathlib import Path

from whatsapp_support.config import RagSettings
from whatsapp_support.mini_app import create_app
from whatsapp_support.web_chat import prepared_manual_directory


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use configured paid provider APIs')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--public-demo', action='store_true',
                        help='Expose only authenticated job endpoints with demo usage limits')
    parser.add_argument('--dataset', choices=['sample', 'emanual'], default='sample')
    parser.add_argument('--knowledge-dir', type=Path, default=Path('knowledge'))
    parser.add_argument('--prepared-root', type=Path, default=Path('.rag-data/prepared-public-v3'))
    parser.add_argument('--embedding-config', default='.rag-data/bailian.local.json')
    parser.add_argument('--response-config', default='.rag-data/deepseek.local.json')
    args = parser.parse_args()
    settings = (RagSettings.from_provider_files(args.embedding_config, args.response_config)
                if args.live else RagSettings.from_env())
    knowledge = (prepared_manual_directory(args.prepared_root)
                 if args.dataset == 'emanual' else args.knowledge_dir)
    # Retain the index convention expected by the existing evaluation scripts.
    settings = replace(settings, index_path=os.environ.get(
        'WHATSAPP_SUPPORT_RAG_INDEX_PATH',
        f'.rag-index/web-chat-{args.dataset}-{settings.mode}.sqlite3'))
    invite = os.environ.get('RAG_DEMO_INVITE') or os.environ.get('WECHAT_DEMO_INVITE', '')
    if len(invite) < 16:
        parser.error('Set RAG_DEMO_INVITE to at least 16 characters')
    app = create_app(settings, invite=invite, knowledge_dir=str(knowledge),
                     dataset=args.dataset, public_demo=args.public_demo)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()

