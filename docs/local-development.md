# Local development

This change is a credential-free, fake-first foundation with contract-level Meta Cloud API webhook and sender adapters. It proves adapter contracts and failure paths without claiming live Meta delivery or requiring any real WhatsApp account, business, knowledge, model, database, or queue integration.

## Deterministic conversation orchestration

The default runtime provides a fake-first conversation orchestration layer. Explicit product requests such as `product sku-1` use `get_product_info`; inventory requests use `get_product_inventory`. The bundled records are deterministic test fixtures and make no HTTP, database, Redis, or provider calls.

Protected customer data remains behind the existing identity scopes: `order:read`, `shipping:read`, and `return:read`. A missing or expired grant prevents the corresponding tool call. Refund, address-change, cancellation, complaint, and human-agent requests are write-risk operations and always use the handoff path. Real business APIs and LLM services are not connected, so local replies must not be treated as live commerce data.

## One-time setup

From the repository root:

```powershell
py -3.12 -m pip install setuptools wheel
py -3.12 -m pip install --no-build-isolation -e .[test]
```

The editable install is needed for `py -3.12 -m whatsapp_support` because the package uses a `src/` layout. Pytest adds `src/` through `tests/conftest.py`, so the test command can run with the available pytest interpreter.

## Local RAG workflow

RAG is also fake-first. The one-time indexing command and worker-time configuration
are deliberately separate: import Markdown/TXT knowledge with
`py -3.12 -m whatsapp_support rag-index knowledge/`, then start a Worker that reads
the completed active generation from `WHATSAPP_SUPPORT_RAG_INDEX_PATH`. Generated
indexes belong under `.rag-index/`; optional full public datasets belong under
`.rag-data/`. Do not remove either directory wholesale.

`WHATSAPP_SUPPORT_RAG_MODE=fake` needs no credential or network. The `openai` mode
requires `OPENAI_API_KEY` plus separately configured `OPENAI_EMBEDDING_MODEL` and
`OPENAI_RESPONSE_MODEL`; the same API key may serve both models. This selection is
independent of in-memory versus durable PostgreSQL/Redis infrastructure.

Static FAQ/product/policy answers use locale-compatible evidence. Live order,
shipping, inventory, or return eligibility stays on authorized tools. Evidence
shortage, hostile output, locale mismatch, write-risk, and `HUMAN_ACTIVE` state use
human handoff or suppress automation. WhatsApp receives one validated full response,
never a stream, and internal evidence is not exposed to customers. Citation-ID
membership and structural validation reduce unsupported answers but do not prove
semantic factuality; evaluation and handoff remain quality controls. Memory keeps
all raw messages while prompts use a rolling summary and the six complete turns
most recently exchanged.

See [rag-local-development.md](rag-local-development.md) for the complete local setup
and safety boundaries.

## Verification

Run the full local checks:

```powershell
py -3.12 -m pytest -q
py -3.12 -m compileall -q src
py -3.12 -m whatsapp_support --help
py -3.12 -m whatsapp_support integration-check
git diff --check
```

The suite has no network or third-party account requirement. It is the default fake-first verification path and tests the domain, in-memory adapters, fake HTTP transport contracts, Meta webhook/sender mappings, worker, knowledge indexer, dispatcher, and HTTP entrypoints. No test sends to Meta or claims live provider delivery.

## WhatsApp provider configuration

The default provider is fake. If `WHATSAPP_SUPPORT_PROVIDER` is unset, runtime selection is equivalent to `fake`; fake mode creates no Meta HTTP client and makes no external request. The exact provider variables are:

- `WHATSAPP_SUPPORT_PROVIDER` (`fake` or `meta_cloud`)
- `WHATSAPP_SUPPORT_META_ACCESS_TOKEN`
- `WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID`
- `WHATSAPP_SUPPORT_META_GRAPH_BASE_URL`

For contract-only Meta wiring, use non-credential placeholders and keep any real values outside terminal history and logs:

```powershell
$env:WHATSAPP_SUPPORT_PROVIDER = 'fake'
# For contract-only Meta wiring, set these outside logs/terminal history:
$env:WHATSAPP_SUPPORT_PROVIDER = 'meta_cloud'
$env:WHATSAPP_SUPPORT_META_ACCESS_TOKEN = '<configure securely; do not print>'
$env:WHATSAPP_SUPPORT_META_PHONE_NUMBER_ID = '<configure securely; do not print>'
$env:WHATSAPP_SUPPORT_META_GRAPH_BASE_URL = '<configure a versioned Graph base URL securely>'
```

`meta_cloud` requires all three Meta values. Incomplete configuration keeps the process constructible but makes provider readiness fail without echoing secret values. Complete Meta readiness validates configuration and transport construction only; it does not send a message. The contract tests inject an in-memory recording/fake transport, so they do not need a token, phone number, real Graph URL, network, or WABA account.

## Optional durable runtime setup

If you want PostgreSQL/Redis-backed verification instead of the fake-first default, install the optional runtime dependencies once:

```powershell
py -3.12 -m pip install --no-build-isolation -e .[runtime,test]
```

Start the local Compose services:

```powershell
docker compose up -d postgres redis
```

Then configure the exact durable environment variable names. Set them in your shell or secret manager, but do not print credential-bearing values into logs, screenshots, or shared reports:

```powershell
$env:WHATSAPP_SUPPORT_DATABASE_URL = '<postgresql+asyncpg URL configured outside logs>'
$env:WHATSAPP_SUPPORT_REDIS_URL = '<redis:// URL configured outside logs>'
$env:WHATSAPP_SUPPORT_REDIS_STREAM = 'whatsapp-support:inbound'
$env:WHATSAPP_SUPPORT_REDIS_GROUP = 'whatsapp-support-workers'
$env:WHATSAPP_SUPPORT_OUTBOX_LEASE_SECONDS = '60'
$env:WHATSAPP_SUPPORT_OUTBOX_MAX_ATTEMPTS = '3'
```

Or set the same exact variable names securely outside shell history/logs:

```text
WHATSAPP_SUPPORT_DATABASE_URL=<set this securely>
WHATSAPP_SUPPORT_REDIS_URL=<set this securely>
```

Run the configured integration verification separately from the default suite:

```powershell
py -3.12 -m pytest -m integration -q
py -3.12 -m whatsapp_support integration-check
```

`pytest -m integration` uses the `integration` marker and skips with an explicit reason when `WHATSAPP_SUPPORT_DATABASE_URL` and `WHATSAPP_SUPPORT_REDIS_URL` are not configured. `integration-check` uses the same `DurableSettings` parsing and `/readyz` dependency probe as runtime startup. With no URLs configured it reports that durable runtime is disabled; with only one URL configured it fails clearly without echoing the configured secret URL; with both configured it verifies PostgreSQL schema creation and Redis stream-group readiness.

## Start local roles

Each command blocks in the foreground and binds only to `127.0.0.1`:

```powershell
py -3.12 -m whatsapp_support gateway --port 8080
py -3.12 -m whatsapp_support worker --port 8081
py -3.12 -m whatsapp_support indexer --port 8082
```

The equivalent installed console scripts are `whatsapp-gateway`, `whatsapp-worker`, and `whatsapp-indexer` when the Python Scripts directory is on `PATH`.

All roles expose:

```text
GET /healthz  -> {"status":"ok","role":"..."}
GET /readyz   -> {"status":"ready","role":"..."}
```

`/healthz` is only a liveness check: HTTP 200 means the process can answer and does not prove provider or durable dependency readiness. `/readyz` is the application readiness check. It stays ready in fake-first mode, checks Meta configuration/transport construction without sending in `meta_cloud` mode, and when durable mode is configured it also depends on the PostgreSQL schema and Redis stream group. If any required check cannot initialize, `/readyz` returns HTTP 503 with `{"status":"not_ready","role":"..."}` while `/healthz` remains healthy.

The gateway additionally exposes:

```text
GET  /webhook?hub.mode=subscribe&hub.verify_token=<configured-placeholder>&hub.challenge=<challenge-placeholder>
POST /webhook
```

The POST route verifies `X-Hub-Signature-256` over the exact raw body before decoding it. The gateway has local development defaults for `WHATSAPP_VERIFY_TOKEN` and `WHATSAPP_APP_SECRET`; override them through a secure local mechanism without printing their values.

## Acceptance mapping

| Acceptance | Evidence |
| --- | --- |
| A1: `recipient_id` validation and durable serialization | `tests/test_domain.py`, `tests/test_serialization.py`, `tests/test_adapters.py`, `tests/test_postgres_adapters.py`, and `tests/test_redis_queue.py` |
| A2: signature-first inbound Meta parsing, UTC identity mapping, and Inbox dedup | `tests/test_meta_webhook.py` parser/signature/dedup contracts |
| A3: batch safety, status/unsupported no-reply behavior, and retryable partial queue failure | `tests/test_meta_webhook.py` batch outcome and unqueued-key contracts |
| A4: Meta text/template JSON, recipient-only routing, and provider ID extraction | `tests/test_meta_whatsapp.py` request/response contracts |
| A5: retryable network/timeout/429/5xx, permanent 4xx, and safe redaction | `tests/test_meta_whatsapp.py` error classification and `tests/test_worker.py` dispatcher transitions |
| A6: fake default, incomplete Meta not-ready, and complete Meta ready without send | `tests/test_config.py` and `tests/test_entrypoints.py` provider/readiness contracts |
| A7: fake-first Inbox/Redis/Conversation/Outbox/durable-worker regression | focused durable command below; seven existing external-service tests skip when PostgreSQL/Redis are not configured |
| A8: repeatable Python 3.12 pytest/compile/help/integration-check with no live delivery | full verification commands below and fake transport contract tests |

The A7 durable regression command is:

```powershell
py -3.12 -m pytest tests/test_adapters.py tests/test_durable_worker.py tests/test_postgres_adapters.py tests/test_redis_queue.py tests/test_outbox_recovery.py tests/integration/test_postgres_runtime.py tests/integration/test_redis_runtime.py -q
```

## Deliberate limits

The local implementation does not claim production readiness. In-memory state disappears on process exit, fake tools return fixture values, no real WABA webhook is registered, and no model or embedding service is called. The Meta adapter is contract-only: real WABA registration, provider delivery, media, interactive messages, catalog/group/voice features, inbound reactions, and template management are non-goals. Production integrations must retain the same HMAC, authorization, trace, idempotency, ordering, retry, and handoff invariants.

Configured durable verification narrows some gaps but does not remove the remaining provider timeout ambiguity: after a send timeout the stable dedup key remains and the record stays retryable, yet an external provider may still have accepted the original delivery. This preserves retryability but leaves an explicit at-least-once duplicate-delivery risk. Outbox lease recovery and Redis reclaim protect replay semantics after restarts, but they do not prove end-to-end provider acknowledgement without a real provider-side confirmation channel.
