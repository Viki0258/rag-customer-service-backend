"""Convert selected public dataset rows to the strict local evaluation schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


SCHEMA_VERSION = "1.0"
CONVERTER_VERSION = "1.0"
SUPPORTED_DATASETS = frozenset(
    {"multidoc2dial", "ragbench-emanual", "bitext-customer-support"}
)
MAX_ZIP_MEMBERS = 128
MAX_ZIP_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200


def _required_text(record: Mapping[str, Any], key: str, *, row_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record {row_number} has empty or missing {key}")
    return value.strip()


def _base_case(
    *,
    case_id: str,
    locale: str,
    question: str,
    task_kind: str,
    evidence_candidates: list[dict[str, str]],
    relevant_ids: list[str],
    source_dataset: str,
) -> dict[str, Any]:
    relevant_id_set = set(relevant_ids)
    return {
        "record_type": "case",
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "locale": locale,
        "question": question,
        "task_kind": task_kind,
        "evidence_candidates": {
            candidate["evidence_id"]: {
                "text": candidate["text"],
                "expected_relevant": candidate["evidence_id"] in relevant_id_set,
            }
            for candidate in evidence_candidates
        },
        "source_dataset": source_dataset,
        "synthetic": False,
    }


def _convert_multidoc2dial(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    raw_id = record.get("id", record.get("case_id"))
    if not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
        raise ValueError(f"record {row_number} has empty or missing id")
    case_id = str(raw_id).strip()
    question = _required_text(record, "question", row_number=row_number)
    locale = record.get("locale", "en")
    if locale not in {"en", "zh"}:
        raise ValueError(f"record {row_number} has unsupported locale")

    documents = record.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"record {row_number} has empty or missing documents")
    evidence: list[dict[str, str]] = []
    known_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError(f"record {row_number} contains an invalid document")
        document_id = document.get("id", document.get("doc_id"))
        if not isinstance(document_id, (str, int)) or not str(document_id).strip():
            raise ValueError(f"record {row_number} contains a document with empty id")
        document_id = str(document_id).strip()
        spans = document.get("spans")
        if isinstance(spans, list) and spans:
            for span in spans:
                if not isinstance(span, Mapping):
                    raise ValueError(f"record {row_number} contains an invalid span")
                span_id = span.get("id", span.get("span_id"))
                text = span.get("text")
                if not isinstance(span_id, (str, int)) or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"record {row_number} contains an empty span")
                evidence_id = f"{document_id}:{str(span_id).strip()}"
                if evidence_id in known_ids:
                    raise ValueError(f"record {row_number} contains duplicate evidence id")
                known_ids.add(evidence_id)
                evidence.append({"evidence_id": evidence_id, "text": text.strip()})
        else:
            text = document.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"record {row_number} contains an empty document")
            if document_id in known_ids:
                raise ValueError(f"record {row_number} contains duplicate evidence id")
            known_ids.add(document_id)
            evidence.append({"evidence_id": document_id, "text": text.strip()})

    grounding = record.get("grounding", record.get("references"))
    if not isinstance(grounding, list) or not grounding:
        raise ValueError(f"record {row_number} has empty or missing grounding")
    relevant_ids: list[str] = []
    for item in grounding:
        if not isinstance(item, Mapping):
            raise ValueError(f"record {row_number} contains invalid grounding")
        document_id = item.get("document_id", item.get("doc_id"))
        span_id = item.get("span_id", item.get("id"))
        if document_id is None:
            raise ValueError(f"record {row_number} grounding has empty document id")
        evidence_id = str(document_id).strip()
        if span_id is not None:
            evidence_id = f"{evidence_id}:{str(span_id).strip()}"
        if evidence_id not in known_ids:
            raise ValueError(f"record {row_number} grounding references unknown evidence id")
        if evidence_id not in relevant_ids:
            relevant_ids.append(evidence_id)

    evidence.sort(key=lambda item: item["evidence_id"])
    relevant_ids.sort()
    return _base_case(
        case_id=f"multidoc2dial:{case_id}",
        locale=locale,
        question=question,
        task_kind="retrieval_answer",
        evidence_candidates=evidence,
        relevant_ids=relevant_ids,
        source_dataset="multidoc2dial",
    )


def _alpha_index(index: int) -> str:
    value = index
    letters = ""
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(ord("a") + remainder) + letters
        if value == 0:
            return letters
        value -= 1


def _ragbench_sentence_items(
    record: Mapping[str, Any], row_number: int, relevant_keys: list[str]
) -> list[tuple[str, str]]:
    raw_sentences = record.get("documents_sentences")
    if isinstance(raw_sentences, list) and raw_sentences:
        # Publisher Parquet uses document -> [sentence key, sentence text].
        if all(isinstance(group, list) and all(
            isinstance(pair, list) and len(pair) == 2
            and all(isinstance(value, str) and value.strip() for value in pair)
            for pair in group
        ) for group in raw_sentences):
            items = [(key.strip(), text.strip()) for group in raw_sentences for key, text in group]
            if not items or len({key for key, _ in items}) != len(items):
                raise ValueError(f"record {row_number} has empty or duplicate sentence keys")
            return items
        grouped: list[list[str]] = []
        for group in raw_sentences:
            values = group if isinstance(group, list) else [group]
            normalized_group: list[str] = []
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"record {row_number} contains an empty document sentence")
                normalized_group.append(value.strip())
            grouped.append(normalized_group)
        flattened = [sentence for group in grouped for sentence in group]
        numeric_keys = bool(relevant_keys) and all(
            key.isdecimal() and int(key) < len(flattened) for key in relevant_keys
        )
        if numeric_keys:
            return [(str(index), text) for index, text in enumerate(flattened)]
        return [
            (f"{document_index}{_alpha_index(sentence_index)}", text)
            for document_index, group in enumerate(grouped)
            for sentence_index, text in enumerate(group)
        ]
    documents = record.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"record {row_number} has empty or missing documents")
    if not all(isinstance(item, str) and item.strip() for item in documents):
        raise ValueError(f"record {row_number} contains an empty document")
    return [(str(index), item.strip()) for index, item in enumerate(documents)]


def _convert_ragbench(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    raw_id = record.get("id")
    if not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
        raise ValueError(f"record {row_number} has empty or missing id")
    record_id = str(raw_id).strip()
    # RAGBench repeats each question ID for different source answer generators.
    # Preserve both annotated records instead of collapsing them into one gold.
    if record.get("generation_model_name"):
        record_id += ':' + _required_text(record, "generation_model_name", row_number=row_number)
    question = _required_text(record, "question", row_number=row_number)
    relevant_keys = record.get("all_relevant_sentence_keys", [])
    unsupported_keys = record.get("unsupported_response_sentence_keys", [])
    support_info = record.get("sentence_support_information", {})
    if isinstance(support_info, list):
        normalized = {}
        for item in support_info:
            if not isinstance(item, Mapping):
                raise ValueError(f"record {row_number} has invalid sentence support item")
            key = _required_text(item, "response_sentence_key", row_number=row_number)
            if key in normalized:
                raise ValueError(f"record {row_number} has duplicate response sentence key")
            normalized[key] = item.get("supporting_sentence_keys")
        support_info = normalized
    if not isinstance(relevant_keys, list) or not isinstance(unsupported_keys, list):
        raise ValueError(f"record {row_number} has invalid support annotations")
    if not isinstance(support_info, Mapping):
        raise ValueError(f"record {row_number} has invalid sentence_support_information")
    normalized_relevant = [str(key) for key in relevant_keys]
    normalized_unsupported = [str(key) for key in unsupported_keys]
    normalized_support_info: dict[str, list[str]] = {}
    for key, values in support_info.items():
        if not isinstance(values, list):
            raise ValueError(f"record {row_number} has invalid sentence_support_information")
        normalized_support_info[str(key)] = [str(value) for value in values]

    sentence_items = _ragbench_sentence_items(record, row_number, normalized_relevant)
    sentence_keys = {key for key, _text in sentence_items}
    evidence = [
        {"evidence_id": f"{record_id}:sentence:{key}", "text": text}
        for key, text in sentence_items
    ]
    relevant_ids: list[str] = []
    for key in normalized_relevant:
        if key not in sentence_keys:
            raise ValueError(f"record {row_number} relevant sentence key is not a local sentence key")
        relevant_ids.append(f"{record_id}:sentence:{key}")

    response_unit_keys = sorted(
        set(normalized_unsupported) | set(normalized_support_info),
        key=lambda value: (not value.isdecimal(), int(value) if value.isdecimal() else value),
    )
    support_labels = {
        key: "unsupported" if key in normalized_unsupported else "supported"
        for key in response_unit_keys
    }
    adherence_score = record.get("adherence_score")
    if isinstance(adherence_score, bool):
        adherence_score = float(adherence_score)
    if not isinstance(adherence_score, (int, float)) or isinstance(adherence_score, bool):
        raise ValueError(f"record {row_number} has invalid adherence_score")

    case = _base_case(
        case_id=f"ragbench-emanual:{record_id}",
        locale="en",
        question=question,
        task_kind="retrieval_answer",
        evidence_candidates=evidence,
        relevant_ids=relevant_ids,
        source_dataset="ragbench-emanual",
    )
    case["support_labels"] = support_labels
    case["source_annotations"] = {
        "adherence_score": float(adherence_score),
        "all_relevant_sentence_keys": normalized_relevant,
        "sentence_support_information": normalized_support_info,
        "unsupported_response_sentence_keys": normalized_unsupported,
    }
    return case


def _convert_bitext(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    question = _required_text(record, "instruction", row_number=row_number)
    intent = _required_text(record, "intent", row_number=row_number)
    content_id = hashlib.sha256(f"{intent}\0{question}".encode("utf-8")).hexdigest()
    return {
        "record_type": "case",
        "schema_version": SCHEMA_VERSION,
        "case_id": f"bitext-customer-support:{content_id}",
        "locale": "en",
        "question": question,
        "task_kind": "routing_only",
        "evidence_candidates": {},
        "expected_intent": intent,
        "source_dataset": "bitext-customer-support",
        "synthetic": False,
    }


def convert_records(dataset: str, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert records for exactly one supported dataset without I/O side effects."""

    converters = {
        "multidoc2dial": _convert_multidoc2dial,
        "ragbench-emanual": _convert_ragbench,
        "bitext-customer-support": _convert_bitext,
    }
    try:
        converter = converters[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc

    converted: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for row_number, record in enumerate(records):
        if not isinstance(record, Mapping) or not record:
            raise ValueError(f"record {row_number} is empty or invalid")
        case = converter(record, row_number)
        case_id = case["case_id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate output case_id: {case_id}")
        case_ids.add(case_id)
        converted.append(case)
    if not converted:
        raise ValueError("input contains no records")
    return sorted(converted, key=lambda row: row["case_id"])


def _validate_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"ZIP contains more than {MAX_ZIP_MEMBERS} members")
    total_size = 0
    for member in members:
        normalized_name = member.filename.replace("\\", "/")
        path = PurePosixPath(normalized_name)
        if (
            normalized_name != member.filename
            or path.is_absolute()
            or ".." in path.parts
            or (path.parts and ":" in path.parts[0])
        ):
            raise ValueError(f"unsafe ZIP member path: {member.filename}")
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise ValueError(f"unsafe ZIP member link: {member.filename}")
        if member.flag_bits & 0x1:
            raise ValueError(f"encrypted ZIP member is unsupported: {member.filename}")
        if member.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ValueError(f"ZIP member exceeds size limit: {member.filename}")
        total_size += member.file_size
        if total_size > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("ZIP exceeds total uncompressed size limit")
        if member.file_size >= 64 * 1024:
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > MAX_ZIP_COMPRESSION_RATIO:
                raise ValueError(f"ZIP member exceeds compression ratio limit: {member.filename}")
    return members


def _read_zip_json(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> Mapping[str, Any]:
    try:
        with archive.open(member, "r") as stream:
            payload = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON in ZIP member {member.filename}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"ZIP member {member.filename} must contain an object")
    return payload


def _document_spans(
    document: Mapping[str, Any], *, document_id: str, member_name: str
) -> list[dict[str, str]]:
    raw_spans = document.get("spans")
    span_values = raw_spans.values() if isinstance(raw_spans, Mapping) else raw_spans
    if not isinstance(span_values, Iterable):
        raise ValueError(f"document {document_id} in {member_name} has no spans")
    spans: list[dict[str, str]] = []
    for span in span_values:
        if not isinstance(span, Mapping):
            raise ValueError(f"document {document_id} in {member_name} has invalid span")
        span_id = span.get("id_sp", span.get("span_id", span.get("id")))
        text = span.get("text_sp", span.get("text"))
        if not isinstance(span_id, (str, int)) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"document {document_id} in {member_name} has empty span")
        spans.append({"id": str(span_id).strip(), "text": text.strip()})
    if not spans:
        raise ValueError(f"document {document_id} in {member_name} has no spans")
    return sorted(spans, key=lambda item: item["id"])


def _multidoc_records(
    documents_payload: Mapping[str, Any],
    dialogue_payloads: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    raw_doc_data = documents_payload.get("doc_data")
    if not isinstance(raw_doc_data, Mapping):
        raise ValueError("MultiDoc2Dial document JSON has no doc_data object")
    documents_by_domain: dict[str, dict[str, Mapping[str, Any]]] = {}
    for domain, raw_documents in raw_doc_data.items():
        if not isinstance(domain, str) or not isinstance(raw_documents, Mapping):
            raise ValueError("MultiDoc2Dial doc_data has invalid domain records")
        indexed_documents: dict[str, Mapping[str, Any]] = {}
        for source_key, document in raw_documents.items():
            if not isinstance(document, Mapping):
                raise ValueError(f"MultiDoc2Dial domain {domain} contains an invalid document")
            document_id = document.get("doc_id", source_key)
            if not isinstance(document_id, (str, int)) or not str(document_id).strip():
                raise ValueError(f"MultiDoc2Dial domain {domain} contains a document without doc_id")
            normalized_id = str(document_id).strip()
            if normalized_id in indexed_documents:
                raise ValueError(f"MultiDoc2Dial domain {domain} contains duplicate doc_id")
            indexed_documents[normalized_id] = document
        documents_by_domain[domain] = indexed_documents

    records: list[Mapping[str, Any]] = []
    for member_name, payload in sorted(dialogue_payloads):
        raw_dial_data = payload.get("dial_data")
        if not isinstance(raw_dial_data, Mapping):
            raise ValueError(f"{member_name} has no dial_data object")
        for domain in sorted(raw_dial_data):
            dialogues = raw_dial_data[domain]
            domain_documents = documents_by_domain.get(str(domain))
            if not isinstance(dialogues, list) or domain_documents is None:
                raise ValueError(f"{member_name} has invalid or unknown domain {domain}")
            for dialogue in sorted(dialogues, key=lambda item: str(item.get("dial_id", ""))):
                if not isinstance(dialogue, Mapping):
                    raise ValueError(f"{member_name} contains an invalid dialogue")
                dial_id = dialogue.get("dial_id")
                turns = dialogue.get("turns")
                if not isinstance(dial_id, (str, int)) or not isinstance(turns, list):
                    raise ValueError(f"{member_name} contains a dialogue without ID or turns")
                ordered_turns = sorted(turns, key=lambda turn: int(turn.get("turn_id", 0)))
                dialogue_doc_ids = sorted(
                    {
                        str(reference.get("doc_id"))
                        for turn in ordered_turns
                        if isinstance(turn, Mapping)
                        for reference in turn.get("references", [])
                        if isinstance(reference, Mapping) and reference.get("doc_id") is not None
                    }
                )
                normalized_documents = []
                for document_id in dialogue_doc_ids:
                    document = domain_documents.get(document_id)
                    if document is None:
                        raise ValueError(f"dialogue {dial_id} references unknown document {document_id}")
                    normalized_documents.append(
                        {
                            "id": document_id,
                            "spans": _document_spans(
                                document,
                                document_id=document_id,
                                member_name=member_name,
                            ),
                        }
                    )

                previous_user_question: str | None = None
                for turn in ordered_turns:
                    if not isinstance(turn, Mapping):
                        raise ValueError(f"dialogue {dial_id} contains an invalid turn")
                    role = turn.get("role")
                    utterance = turn.get("utterance")
                    if role == "user":
                        previous_user_question = utterance.strip() if isinstance(utterance, str) else None
                        continue
                    references = turn.get("references")
                    if role != "agent" or not isinstance(references, list) or not references:
                        continue
                    if not previous_user_question:
                        raise ValueError(f"grounded agent turn in dialogue {dial_id} has no user question")
                    turn_id = turn.get("turn_id")
                    grounding = []
                    for reference in references:
                        if not isinstance(reference, Mapping):
                            raise ValueError(f"dialogue {dial_id} contains invalid grounding")
                        document_id = reference.get("doc_id")
                        span_id = reference.get("id_sp", reference.get("span_id"))
                        if document_id is None or span_id is None:
                            raise ValueError(f"dialogue {dial_id} contains incomplete grounding")
                        grounding.append(
                            {"document_id": str(document_id), "span_id": str(span_id)}
                        )
                    records.append(
                        {
                            "id": f"{dial_id}:{turn_id}",
                            "question": previous_user_question,
                            "locale": "en",
                            "documents": normalized_documents,
                            "grounding": grounding,
                        }
                    )
    if not records:
        raise ValueError("MultiDoc2Dial archive contains no grounded dialogue turns")
    return records


def _read_multidoc_archive(path: Path) -> list[Mapping[str, Any]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = _validate_zip_members(archive)
            document_members = [
                member
                for member in members
                if PurePosixPath(member.filename).name == "multidoc2dial_doc.json"
            ]
            dialogue_members = [
                member
                for member in members
                if PurePosixPath(member.filename).name.startswith("multidoc2dial_dial_")
                and member.filename.endswith(".json")
            ]
            if len(document_members) != 1 or not dialogue_members:
                raise ValueError(
                    "MultiDoc2Dial ZIP requires one multidoc2dial_doc.json and dialogue JSON files"
                )
            documents = _read_zip_json(archive, document_members[0])
            dialogues = [
                (member.filename, _read_zip_json(archive, member))
                for member in dialogue_members
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {path}") from exc
    return _multidoc_records(documents, dialogues)


def _read_json_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("records", "data", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"JSON input {path} must contain a record list")


def _read_jsonl_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"empty JSONL record at line {line_number}")
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL at line {line_number}: {exc}") from exc
    return records


def _read_parquet(path: Path) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet conversion requires the optional evaluation dependency; "
            "run `pip install pyarrow`."
        ) from exc
    return parquet.read_table(path).to_pylist()


def read_records(
    path: Path | str, *, dataset: str | None = None
) -> list[Mapping[str, Any]]:
    """Read a local JSON, JSONL, CSV, or optionally Parquet record file."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".zip":
        if dataset != "multidoc2dial":
            raise ValueError("ZIP input is supported only for multidoc2dial")
        return _read_multidoc_archive(source)
    if suffix == ".json":
        return _read_json_records(source)
    if suffix == ".jsonl":
        return _read_jsonl_records(source)
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".parquet":
        return _read_parquet(source)
    raise ValueError(f"unsupported input format: {suffix or '<none>'}")


def write_jsonl(rows: Sequence[Mapping[str, Any]], destination: Path | str) -> None:
    """Atomically write deterministic UTF-8 JSONL output."""

    final_path = Path(destination).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{final_path.name}.",
            suffix=".part",
            dir=final_path.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                output.write("\n")
        temp_path.replace(final_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    converted = convert_records(
        args.dataset,
        read_records(args.source, dataset=args.dataset),
    )
    write_jsonl(converted, args.destination)
    print(f"wrote {len(converted)} records to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
