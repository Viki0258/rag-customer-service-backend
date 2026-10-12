# RAG local development

The RAG subsystem is fake-first and can be exercised locally without a network,
OpenAI account, WhatsApp account, PostgreSQL, or Redis. Local acceptance tests use
temporary knowledge files, a real SQLite knowledge index, deterministic fake
embedding/answer models, and a SQLAlchemy SQLite database. They do not prove live
provider or production integration.

## Configuration boundaries

Indexing and message handling are separate operations. Import-time configuration
is consumed by the one-time indexing command; worker-time configuration is consumed
when a Worker opens the completed active generation. Do not run indexing in the
message-handling path.

The shared settings are:

- `WHATSAPP_SUPPORT_RAG_MODE`: `fake` or `openai`.
- `WHATSAPP_SUPPORT_RAG_INDEX_PATH`: the SQLite index file, normally below
  `.rag-index/`.
- `WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION`: the configured embedding dimension.
- `OPENAI_EMBEDDING_MODEL`: the embedding model selected by configuration.
- `OPENAI_RESPONSE_MODEL`: the answer model selected by configuration.
- `OPENAI_API_KEY`: required when `WHATSAPP_SUPPORT_RAG_MODE=openai`.

The same API key can serve the separately configured embedding and answer models.
Their names, dimensions, timeouts, and retry limits are independent. Model
availability changes over time, so this guide intentionally gives no hardcoded
model selection. Durable PostgreSQL/Redis infrastructure is also independent of
model selection: fake RAG can use durable storage, and OpenAI RAG can use local
in-memory conversation storage.

In `fake` mode, runtime construction creates no OpenAI HTTP client and sends no
network request. In `openai` mode, all required OpenAI values must be provided
securely outside source control, shell transcripts, logs, and reports. The normal
test suite continues to use local fakes; explicitly enabled external integration
tests remain separate and skipped when their configuration is absent.

## Build a local knowledge index

### Bailian embeddings with DeepSeek replies

Set `WHATSAPP_SUPPORT_RAG_MODE=compatible` to use independently configured
OpenAI-compatible embedding and Responses endpoints. This mode explicitly loads
two local JSON files; fake mode does not read them or contact either provider.

- `.rag-data/bailian.local.json`: `api_key`, `base_url`,
  `embedding_model` (`text-embedding-v4`), and `dimensions` (`1024`).
- `.rag-data/deepseek.local.json`: `api_key`, `base_url`
  (`https://api.deepseek.com`), and `response_model` (`deepseek-flash`).

The current DeepSeek-V4.1-Flash API ID is `deepseek-flash`, as documented in
[the official release](https://api-docs.deepseek.com/news/news260910/).
The adapter uses [DeepSeek Responses](https://api-docs.deepseek.com/guides/responses_api/)
with the existing strict JSON schema and full-response validation. API format
support was verified with the local smoke check described below. Other accounts
still require their own valid key and integration test.

These files contain plaintext credentials and must remain local. `.rag-data/` is
Git-ignored; never force-add these files or paste their contents into logs.
Override the file paths with `WHATSAPP_SUPPORT_EMBEDDING_CONFIG_FILE` and
`WHATSAPP_SUPPORT_RESPONSE_CONFIG_FILE` if needed. Base URLs must use HTTPS and
must not contain credentials, query parameters, or fragments. Copy Bailian's
OpenAI-compatible base URL from the same account/region as its key.

Install the existing runtime extra before real HTTP calls:

```powershell
py -3.12 -m pip install -e ".[runtime]"
$env:WHATSAPP_SUPPORT_RAG_MODE = 'compatible'
py -3.12 -m whatsapp_support rag-index knowledge/
```

Run commands from the repository root. Both provider files must be complete in
this mode, including for indexing. The default index is `.rag-index/bailian.sqlite3`,
separate from the old fake index; `WHATSAPP_SUPPORT_RAG_INDEX_PATH` overrides it.
Embedding requests are sent in batches of at most ten and retain input ordering.
The reply timeout defaults to 90 seconds with zero automatic retries in this mode;
existing timeout/retry and retrieval/memory environment settings remain available.
Neither CLI indexing nor configuration loading sends WhatsApp messages.

Create a `knowledge/` directory containing only Markdown or TXT files. Include the
locale in each filename, for example `returns.en.md` and `returns.zh.md`. Then set an
ignored destination and run the one-time indexing command:

```powershell
$env:WHATSAPP_SUPPORT_RAG_MODE = 'fake'
$env:WHATSAPP_SUPPORT_RAG_INDEX_PATH = '.rag-index/knowledge.sqlite3'
$env:WHATSAPP_SUPPORT_RAG_EMBEDDING_DIMENSION = '2'
$env:OPENAI_EMBEDDING_MODEL = '<embedding model configured for this index>'
py -3.12 -m whatsapp_support rag-index knowledge/
```

The command builds and validates a new SQLite generation before atomically making
it active. A failed import leaves the prior active generation unchanged. Rebuild
when the configured embedding model or dimension changes. Workers read only the
active generation; indexing is not repeated for each message.

`.rag-index/` stores local generated indexes. `.rag-data/` stores optional complete
public datasets downloaded by the opt-in dataset tooling. Both are ignored by Git.
They may contain valuable user data, so inspect them and remove only known generated
test artifacts; never delete either directory wholesale.

## Locale, routing, and handoff

Retrieval normalizes locale and accepts an exact locale or a compatible base-language
tag such as `zh` and `zh-CN`. It does not fall back across languages. A locale with
no compatible evidence produces human handoff rather than an answer.

Static FAQ, product descriptions, delivery policy, and return policy use RAG. Live
order, shipping, inventory, or return eligibility remains on the authenticated,
read-only business tools; RAG is not a source for those facts. Refunds, address
changes, cancellations, complaints, and explicit agent requests are write-risk and
go directly to human handoff. Insufficient evidence, model failure, invalid output,
locale mismatch, or an active human conversation also prevents an automated reply.

Conversation memory retains all raw messages for audit and reconstruction. Prompt
context uses a rolling summary plus the six complete turns most recently exchanged.
At the summary threshold, coverage and version advance without deleting raw rows.
The summary helps interpret follow-up questions but is never evidence for policy,
price, inventory, order, or shipping facts.

## Grounding and customer-visible output

Knowledge text is untrusted data. Instructions inside a document cannot override
system rules, call tools, expose prompts, or authorize writes. Model output must be
structured, use evidence identifiers from the current retrieval set, match the
conversation locale, and pass safety and length validation.

Citation-ID membership and structural validation reduce unsupported answers but do
not prove semantic factuality. Offline evaluation and human handoff remain required
quality controls. Internal evidence IDs, rankings, prompts, and audit metadata are
retained for diagnostics; evidence is not exposed to customers.

WhatsApp delivery is full-response-only: the complete response is generated and
validated before exactly one Outbox command is committed. There is no streaming or
partial customer send. The Outbox command carries a stable trace ID and dedup key;
replaying an already processed inbound event creates neither a second message nor a
second command.

## Local verification

### Isolated PostgreSQL/Redis integration (Windows)

Start Docker Desktop and install the project's runtime/test extras, then run:

```powershell
pwsh -NoProfile -File scripts/test-isolated-integration.ps1
```

The PostgreSQL integration fixtures truncate tables. Never run them against a real
business database. This runner creates a uniquely named Compose project from
`compose.integration.yaml`, with random loopback-only ports and tmpfs storage.
It overrides inherited database/Redis URLs only for the process, forces fake RAG,
runs `tests/integration`, and tears down only its own services in `finally`.
All synthetic test data is intentionally discarded. No real model calls or WhatsApp
sends occur. Original environment values are restored. Docker Desktop itself is
not stopped, and existing containers/databases are not targeted.

If forcibly interrupted or the host crashes before cleanup, inspect the printed
`rag-integration-<uuid>` project and remove only that disposable project manually;
do not use global Docker prune commands. Downloaded images remain cached locally.

### Live provider smoke check (2026-09-21)

The production HTTP adapters were exercised against Bailian `text-embedding-v4`
(1024 dimensions) and DeepSeek `deepseek-flash` using a separate local SQLite
index built from the repository's sample `knowledge/` documents. All five embedding
requests and three non-streaming Responses requests returned HTTP 200. English
shipping and Chinese return-policy questions produced validated replies; a solar
panel warranty question absent from the samples produced a handoff result.

Provider-reported usage for this run was 192 embedding tokens and 2,491 response
tokens (1,860 input and 631 output). Responses requests took approximately
1.2–2.6 seconds, excluding retrieval. The Chinese reply was unnecessarily verbose;
this remains a business-evaluation item. No WhatsApp message was sent. This is
connectivity and basic behavior evidence, not a business-quality acceptance test.
Credentials, generated index, and raw report remain in ignored local storage.

When another editable checkout of `whatsapp_support` is installed, select this
repository's source explicitly before local checks to avoid importing that checkout:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

Run the credential-free suite and packaging checks from the repository root:

```powershell
py -3.12 -m pytest
py -3.12 -m whatsapp_support integration-check
git diff --check
```

The wheel build used for final verification must load `build` only from the ignored
SDD `build-tools` directory through a process-only `PYTHONPATH`, write to ignored
scratch output, and inspect the wheel archive and metadata without installing it.
