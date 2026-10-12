# RAG datasets and offline evaluation

## Executed public-data preparation

The real source downloads and conversions were executed on 2026-09-22.
See [preparation results](public-data-preparation-2026-09-22.md) for actual counts,
source defects, exclusions, and the local knowledge corpus locations.
This is data preparation, not a live-model quality evaluation.

After downloading all three sources to `.rag-data/public/`, run:

```powershell
py -3.12 -m scripts.datasets.prepare_public --output .rag-data/prepared-public-v3
```

The output directory must not exist. The pipeline checks recorded download hashes,
keeps dataset/split boundaries, exports source documents only, validates evaluation
rows, deduplicates Bitext intent/instruction pairs, and records source annotation
errors in quarantine. Only a report with `status: complete` is a completed run.

The dataset tools are offline by default. Normal tests use hand-authored fixtures and
replace the downloader's byte-stream boundary; they do not contact a model, a dataset
host, or any other network service. Full public data belongs only in the ignored
repository-root `.rag-data/` directory and must never be committed.

## Public dataset identities

The project supports exactly these publisher-owned sources:

| CLI name | Publisher source | License | Selection | Evaluation use |
| --- | --- | --- | --- | --- |
| `multidoc2dial` | [IBM MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | Apache-2.0 | Published MultiDoc2Dial data, all splits in the archive | Retrieval cases; only explicit grounding document/span IDs are gold |
| `ragbench-emanual` | [Galileo RAGBench dataset card](https://huggingface.co/datasets/galileo-ai/ragbench/raw/main/README.md) | CC BY 4.0 | `emanual`, `test`, revision `97808f3e5fd16ede40bbff6c2949af8139b2eb7b` | Retrieval and annotated answer support; the source response is not a gold grounded answer |
| `bitext-customer-support` | [Bitext dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/blob/main/README.md) | CDLA-Sharing-1.0 | default configuration, `train`, revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443` | English intent/utterance variation only; never factual retrieval gold |

The downloader records the source URL, selected configuration and split, license,
retrieval time, schema/converter versions, configured download URL, effective response
URL, source revision when the approved endpoint provides one, and a SHA-256 of the exact
downloaded bytes. The Hugging Face artifact URLs are pinned to the revisions above.
MultiDoc2Dial has no approved immutable revision, so its downloaded SHA-256 is the
immutable content identity.
The local destination name is always supplied by the operator; remote filename headers
are never followed.

## Opt-in download

From the repository root, choose an explicit supported dataset and a destination under
`.rag-data/`:

```powershell
py -3.12 scripts/datasets/download_public.py ragbench-emanual .rag-data/emanual-test.parquet --allow-network
```

Without `--allow-network`, the command exits before opening the network boundary. The
checkout root is discovered from the tool's installed path; cwd and a caller-provided
repository-root option cannot redirect it.
Destinations that resolve outside this repository's `.rag-data/` are rejected,
including traversal, absolute siblings, prefix-lookalikes, and escaping symlinks or
Windows junctions. A successful download atomically publishes the data and a neighboring
`<filename>.metadata.json`. An existing file is reused only when its recorded SHA-256
and complete authoritative identity (dataset, source/download URL, license,
split/config, revision, and schema/converter version) match the current request;
otherwise it is not overwritten. Streaming stops at a 1 GiB response limit, and an
oversized or interrupted response leaves neither final file nor partial files behind.
Publication uses a neighboring content-hashed transaction journal. If the process
terminates between publishing metadata and data, the next invocation accepts a complete
matching pair or removes only incomplete artifacts whose hashes match that journal.
Malformed journals or mismatched final files stop recovery without deleting them.

## Conversion

Conversion is local and deterministic:

```powershell
py -3.12 scripts/datasets/convert_public.py ragbench-emanual .rag-data/emanual-test.parquet .rag-data/emanual-test.jsonl
```

JSON, JSONL, and CSV inputs use the Python standard library. Parquet input uses the
declared `evaluation` extra:

```powershell
py -3.12 -m pip install -e ".[evaluation]"
```

If `pyarrow` is absent, the converter fails with an explicit install instruction; it
never returns an empty conversion. Empty and malformed source records are errors.

The converters intentionally preserve dataset boundaries:

- MultiDoc2Dial conversion reads the publisher ZIP directly without extracting it.
  It validates member paths, links, encryption, member/total sizes, member count, and
  compression ratio before reading the authoritative `doc_data` and `dial_data` JSON.
  Dialogue-referenced document spans are candidates; each grounded agent turn's own
  `references[]` document/span pairs are its only relevant IDs.
- RAGBench emanual rows preserve normalized response-sentence support labels plus the
  source adherence score, relevant sentence keys, and unsupported response sentence
  keys. The source `response` is deliberately omitted rather than promoted to a gold
  answer.
- Bitext rows become `routing_only` cases with `expected_intent`; they contain no
  evidence candidates or retrieval gold. Their IDs are SHA-256 values derived from the
  normalized intent and instruction, so IDs do not change when source rows are reordered.

All converter output is sorted by stable `case_id`, and duplicate output IDs are errors.
For published RAGBench records the source answer-generator name is included in the
case ID: one source question has separately annotated model-answer variants. The
preparation wrapper quarantines malformed source gold instead of guessing repairs.
The low-level Bitext converter remains strict; the preparation wrapper explicitly
deduplicates its projected intent/instruction pairs and reports the removed count.

## Strict JSONL contract

[`rag-evaluation-schema.json`](rag-evaluation-schema.json) defines version `1.0` case
and result rows. Both reject unknown fields. Every nonblank physical line must contain
one valid object; blank lines, malformed JSON, unsupported enum values, empty required
or whitespace-only strings, duplicate object keys, and duplicate `case_id` values fail
validation.

Case rows carry a stable ID, `en` or `zh` locale, question, task kind, evidence
candidates, expected relevance, source identity, and synthetic marker. Candidates are
an object keyed by stable evidence ID; every value contains nonblank `text` and an
`expected_relevant` boolean. Expected relevant evidence IDs are therefore exposed as
the keys whose flag is true. This single representation makes candidate-ID uniqueness
and relevance membership structural instead of relying on cross-field validation that
JSON Schema cannot express.
Optional fields hold routing intent, normalized support-label maps keyed by unique unit
ID, and the explicit
RAGBench source annotations. Result rows carry precomputed retrieved IDs and handoff
prediction, with optional generated-reply locale, independently annotated answer
support, citations, and routing intent.

The committed `fixtures/eval/faq_en.jsonl` and `faq_zh.jsonl` are small, hand-authored
ecommerce cases. They are synthetic project fixtures—not excerpts or translations from
any public dataset—and cover grounded answering, insufficient-evidence handoff,
routing-only behavior, and prompt injection.

## Deterministic offline metrics

Evaluate precomputed result rows without creating a model client or downloading data:

```powershell
py -3.12 scripts/eval/run_eval.py fixtures/eval/faq_en.jsonl .rag-data/results-en.jsonl --k 5
```

The JSON report includes schema/converter versions, input counts, every metric's value,
numerator and denominator, and excluded-case counts:

- Recall@k is micro-recall over annotated relevant evidence IDs. Its denominator is the
  number of relevant IDs; cases without retrieval gold are excluded and counted.
- MRR averages the reciprocal rank of the first relevant result over cases with
  retrieval gold, including zero contribution when no relevant item was retrieved.
- Grounded and unsupported answer rates use only generated, answer-eligible rows with
  an explicit `answer_support_label`. Missing support annotations are excluded and
  counted, never treated as success.
- Handoff precision divides correct predicted handoffs by all predicted handoffs. With
  no predicted handoffs, its value and denominator are both `0`.
- Locale consistency applies only to generated replies and compares the annotated
  response locale with the case locale. Cases without a generated reply are excluded
  and counted.

A cited ID that belongs to the evidence candidates proves membership only. Citation
membership does not create semantic-grounding credit; only an independent
`answer_support_label` enters the grounded/unsupported metrics. These fake/offline runs
are tooling checks, not live-model quality claims.
