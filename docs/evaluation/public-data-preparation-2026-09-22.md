# Public data preparation — 2026-09-22

Follow-up: the prepared eManual corpus has since been embedded in the isolated
`.rag-index/web-chat-emanual-compatible.sqlite3` index: 101 active documents,
134 chunks, `text-embedding-v4`, 1024 dimensions. The public-data web test runs at
http://127.0.0.1:8766 using `--live --dataset emanual --port 8766`.
An English USB software-update query returned a validated reply
(`trace-a4ef04d45deb1b054609`), and a refund request handed off without generation
(`trace-2794599f07074b9b73b2`). The USB instructions were checked against source
`e66d2e96bcd55a11c9402bb52485442c4e74e1680e228c3cc3421fd3b0da5d3c.en.txt`.
These are smoke checks, not complete benchmark results. The data-preparation-only
scope below describes the earlier run; the original port 8765 remains a sample server.

Status: downloaded, transformed, validated, and readable by the project's knowledge
loader. No embedding API or generation API was called in this preparation run.
The running web playground still uses the small `knowledge/` fixtures; these data
are not yet embedded or selected by that server.

## Actual results

| Source | Knowledge text | Evaluation records |
| --- | --- | --- |
| IBM MultiDoc2Dial | 488 documents → 475 distinct texts | train 24,603; validation 4,699; test 4,567 |
| Galileo RAGBench eManual test | 396 candidate document occurrences → 101 distinct texts | 132 source answer variants → 130 valid; 2 quarantined |
| Bitext customer support train | None: routing-only, not business facts | 26,872 rows → 24,635 unique intent/instruction pairs; 27 intents |

MultiDoc2Dial source domains: DMV 149, SSA 109, student aid 92, VA 138 documents
before deduplication. These are public-service domains, not an ecommerce product
catalog. They are a retrieval/multi-turn benchmark, not current advice for customers.

The selected eManual test data concern Samsung TV functionality/troubleshooting:
Ambient Mode, SmartThings, channels, picture/sound, network, software updates, etc.
They are retrieved manual excerpts, not a full manual, pricing catalog, inventory,
or verified current product specifications. The 132 rows contain 66 original
question IDs, each with two source answer-generator variants; 130 valid rows must
not be advertised as 130 independent questions.

All three sources are English. No Chinese translation was manufactured. Do not use
these data to claim Chinese retrieval coverage or mix them into store return policy.

## Local artifacts

Root: `.rag-data/prepared-public-v3/` (ignored by Git).

- `report.json`: full counts, licenses, source revisions/download provenance, raw
  checksums and generated evaluation-file checksums.
- `multidoc2dial/knowledge/*.en.txt`: 475 loadable knowledge documents.
- `multidoc2dial/{train,validation,test}.cases.jsonl`: original split boundaries.
- `multidoc2dial/{train,validation,test}.context.jsonl`: preceding dialogue turns,
  domain, split and dialogue IDs for reconstructing multi-turn evaluation.
- `ragbench-emanual/knowledge/*.en.txt`: 101 loadable source excerpts.
- `ragbench-emanual/test.cases.jsonl`: 130 validated answer-support/retrieval cases.
- `ragbench-emanual/quarantine.json`: 2 rejected source annotation records.
- `bitext-customer-support/train.cases.jsonl`: 24,635 routing-only records.
- Each knowledge corpus has an adjacent `documents.manifest.json` mapping exact
  text SHA-256 filenames back to all original document references.

Original files and adjacent download metadata are in `.rag-data/public/`.
Earlier incomplete attempts `prepared-public-v1/` and `prepared-public-v2/` are
retained with `status: preparing`; they must not be used as completed artifacts.
The earlier combined `public/multidoc2dial.cases.jsonl` is superseded by the split
files above. No original source was deleted or overwritten.

## Defects discovered by processing real data

The old converter had passed synthetic tests but did not handle the publisher's
RAGBench Parquet layout: explicit sentence key/text pairs, list-valued sentence
support records, boolean adherence labels, and repeated question IDs across model
variants. The converter now supports these, preserving original sentence IDs and
including generator identity in case IDs.

Two source annotations (`emanual_1` and `emanual_539`, first generator variants)
use gold sentence IDs ending in `.` that do not exist in their document sentence
tables. They are quarantined, not silently repaired or credited. Their source
documents remain in the corpus. Source responses never become knowledge or new
model-answer ground truth.

Bitext has 2,237 repeated normalized intent/instruction pairs. Preparation removes
those projected duplicates while retaining the original CSV. No identical normalized
instruction was found with multiple intents. Source responses are discarded because
they are not verified merchant policy.

## Validation and remaining work

Every exported case was checked against the project's strict case validator.
The actual KnowledgeFileLoader successfully loaded all 576 exported documents with
locale `en`. A dry split at 1,200 characters produced 4,634 MultiDoc2Dial chunks and
134 eManual chunks. This did not construct vectors or query a paid API.

Preparation unit tests cover publisher shapes, source variant IDs, quarantine,
deduplication, corpus provenance, and keeping generated source answers out of knowledge.

Next: use the eManual corpus in an isolated embedding index, connect the web test
entry to that corpus with an explicit English dataset label, then evaluate retrieval
and answer support. Flat case-local sentence IDs are not runtime chunk IDs: map using
source manifests/text offsets before scoring retrieval. Preserve dialogue context for
MultiDoc2Dial and account for repeated RAGBench question variants in aggregate metrics.
No live accuracy, recall, or faithfulness score is claimed here.

## Attribution

- [IBM MultiDoc2Dial](https://github.com/IBM/multidoc2dial), Apache-2.0;
  Feng, Patel, Wan and Joshi, *MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents*, EMNLP 2021.
- [Galileo RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench), CC BY 4.0;
  eManual test revision `97808f3e5fd16ede40bbff6c2949af8139b2eb7b`.
- [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
  CDLA-Sharing-1.0; revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`.

Changes are format conversion, whitespace trimming at text boundaries, exact-text
deduplication, case-ID disambiguation, split/context preservation and defect quarantine.
Full source and derived datasets remain local and are not committed to the repository.
