from __future__ import annotations

import sys

from whatsapp_support.runtime import (
    integration_check_main,
    main,
    rag_index_main,
    role_main,
)


def gateway_main() -> int:
    return role_main("gateway")


def worker_main() -> int:
    return role_main("worker")


def indexer_main() -> int:
    return role_main("indexer")


def knowledge_index_main() -> int:
    return rag_index_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
