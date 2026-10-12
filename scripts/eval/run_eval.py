"""Run deterministic offline RAG evaluation over cases and precomputed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0"
CONVERTER_VERSION = "1.0"
LOCALES = frozenset({"en", "zh"})
TASK_KINDS = frozenset(
    {"retrieval_answer", "handoff", "routing_only", "prompt_injection"}
)
SUPPORT_LABELS = frozenset({"supported", "unsupported"})

CASE_REQUIRED = frozenset(
    {
        "record_type",
        "schema_version",
        "case_id",
        "locale",
        "question",
        "task_kind",
        "evidence_candidates",
        "source_dataset",
        "synthetic",
    }
)
CASE_OPTIONAL = frozenset(
    {"expected_intent", "support_labels", "source_annotations"}
)
RESULT_REQUIRED = frozenset(
    {
        "record_type",
        "schema_version",
        "case_id",
        "retrieved_evidence_ids",
        "predicted_handoff",
    }
)
RESULT_OPTIONAL = frozenset(
    {
        "generated_reply",
        "response_locale",
        "answer_support_label",
        "cited_evidence_ids",
        "predicted_intent",
    }
)


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return value


def _reject_unknown(row: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate object key: {key}")
        value[key] = item
    return value


def validate_case(row: Mapping[str, Any]) -> None:
    _reject_unknown(row, CASE_REQUIRED | CASE_OPTIONAL)
    missing = CASE_REQUIRED - set(row)
    if missing:
        raise ValueError(f"missing case fields: {', '.join(sorted(missing))}")
    if row["record_type"] != "case":
        raise ValueError("record_type must be case")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _nonempty_text(row["case_id"], "case_id")
    _nonempty_text(row["question"], "question")
    _nonempty_text(row["source_dataset"], "source_dataset")
    if row["locale"] not in LOCALES:
        raise ValueError(f"locale must be one of {sorted(LOCALES)}")
    if row["task_kind"] not in TASK_KINDS:
        raise ValueError(f"task_kind must be one of {sorted(TASK_KINDS)}")
    if not isinstance(row["synthetic"], bool):
        raise ValueError("synthetic must be a boolean")

    evidence = row["evidence_candidates"]
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence_candidates must be an object")
    for evidence_id, candidate in evidence.items():
        _nonempty_text(evidence_id, "evidence_id")
        if not isinstance(candidate, Mapping):
            raise ValueError("evidence candidate must be an object")
        _reject_unknown(candidate, frozenset({"text", "expected_relevant"}))
        if set(candidate) != {"text", "expected_relevant"}:
            raise ValueError(
                "evidence candidate requires text and expected_relevant"
            )
        _nonempty_text(candidate["text"], "evidence text")
        if not isinstance(candidate["expected_relevant"], bool):
            raise ValueError("expected_relevant must be a boolean")

    if "expected_intent" in row:
        _nonempty_text(row["expected_intent"], "expected_intent")
    if "support_labels" in row:
        labels = row["support_labels"]
        if not isinstance(labels, Mapping):
            raise ValueError("support_labels must be an object")
        for unit_id, label in labels.items():
            _nonempty_text(unit_id, "support unit_id")
            if label not in SUPPORT_LABELS:
                raise ValueError("support label must be supported or unsupported")
    if "source_annotations" in row:
        annotations = row["source_annotations"]
        if not isinstance(annotations, Mapping):
            raise ValueError("source_annotations must be an object")
        allowed = frozenset(
            {
                "adherence_score",
                "all_relevant_sentence_keys",
                "sentence_support_information",
                "unsupported_response_sentence_keys",
            }
        )
        _reject_unknown(annotations, allowed)
        if set(annotations) != allowed:
            raise ValueError("source_annotations is missing required annotation fields")
        score = annotations["adherence_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("adherence_score must be a number")
        _string_list(annotations["all_relevant_sentence_keys"], "all_relevant_sentence_keys")
        support_information = annotations["sentence_support_information"]
        if not isinstance(support_information, Mapping):
            raise ValueError("sentence_support_information must be an object")
        for unit_id, supporting_ids in support_information.items():
            _nonempty_text(unit_id, "sentence_support_information unit ID")
            _string_list(supporting_ids, "sentence_support_information support IDs")
        _string_list(
            annotations["unsupported_response_sentence_keys"],
            "unsupported_response_sentence_keys",
        )


def validate_result(row: Mapping[str, Any]) -> None:
    _reject_unknown(row, RESULT_REQUIRED | RESULT_OPTIONAL)
    missing = RESULT_REQUIRED - set(row)
    if missing:
        raise ValueError(f"missing result fields: {', '.join(sorted(missing))}")
    if row["record_type"] != "result":
        raise ValueError("record_type must be result")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _nonempty_text(row["case_id"], "case_id")
    _string_list(row["retrieved_evidence_ids"], "retrieved_evidence_ids")
    if not isinstance(row["predicted_handoff"], bool):
        raise ValueError("predicted_handoff must be a boolean")
    if "generated_reply" in row:
        _nonempty_text(row["generated_reply"], "generated_reply")
        if "response_locale" not in row:
            raise ValueError("generated replies require response_locale")
    elif "response_locale" in row or "answer_support_label" in row:
        raise ValueError("response annotations require generated_reply")
    if "response_locale" in row and row["response_locale"] not in LOCALES:
        raise ValueError(f"response_locale must be one of {sorted(LOCALES)}")
    if "answer_support_label" in row and row["answer_support_label"] not in SUPPORT_LABELS:
        raise ValueError("answer_support_label must be supported or unsupported")
    if "cited_evidence_ids" in row:
        _string_list(row["cited_evidence_ids"], "cited_evidence_ids")
    if "predicted_intent" in row:
        _nonempty_text(row["predicted_intent"], "predicted_intent")


def load_jsonl(path: Path | str, *, expected_record_type: str) -> list[dict[str, Any]]:
    """Load every JSONL row strictly; malformed and empty rows are errors."""

    if expected_record_type not in {"case", "result"}:
        raise ValueError("expected_record_type must be case or result")
    validator = validate_case if expected_record_type == "case" else validate_result
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source = Path(path)
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"empty record at line {line_number}")
        try:
            row = json.loads(
                raw_line,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON at line {line_number}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"record at line {line_number} must be an object")
        try:
            validator(row)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        case_id = row["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id at line {line_number}: {case_id}")
        seen_ids.add(case_id)
        records.append(row)
    if not records:
        raise ValueError(f"empty JSONL file: {source}")
    return records


def _rate(numerator: int | float, denominator: int) -> dict[str, int | float]:
    return {
        "value": numerator / denominator if denominator else 0.0,
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate(
    cases: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    """Calculate offline metrics with explicit populations and exclusions."""

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    case_rows = [dict(row) for row in cases]
    result_rows = [dict(row) for row in results]
    case_by_id: dict[str, dict[str, Any]] = {}
    for row in case_rows:
        validate_case(row)
        if row["case_id"] in case_by_id:
            raise ValueError(f"duplicate case_id: {row['case_id']}")
        case_by_id[row["case_id"]] = row
    result_by_id: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        validate_result(row)
        case_id = row["case_id"]
        if case_id in result_by_id:
            raise ValueError(f"duplicate case_id: {case_id}")
        if case_id not in case_by_id:
            raise ValueError(f"result references unknown case_id: {case_id}")
        result_by_id[case_id] = row

    retrieval_without_gold = 0
    relevant_retrieved = 0
    relevant_total = 0
    reciprocal_rank_total = 0.0
    retrieval_case_count = 0
    annotated_answers = 0
    grounded_answers = 0
    unsupported_answers = 0
    answer_support_unannotated = 0
    predicted_handoffs = 0
    correct_handoffs = 0
    generated_replies = 0
    locale_matches = 0
    without_generated_reply = 0

    for case_id, case in case_by_id.items():
        result = result_by_id.get(case_id)
        relevant = [
            evidence_id
            for evidence_id, candidate in case["evidence_candidates"].items()
            if candidate["expected_relevant"]
        ]
        retrieved = result["retrieved_evidence_ids"] if result else []
        if relevant:
            retrieval_case_count += 1
            relevant_total += len(relevant)
            relevant_set = set(relevant)
            top_k = retrieved[:k]
            relevant_retrieved += len(relevant_set.intersection(top_k))
            first_rank = next(
                (rank for rank, evidence_id in enumerate(retrieved, 1) if evidence_id in relevant_set),
                None,
            )
            if first_rank is not None:
                reciprocal_rank_total += 1 / first_rank
        else:
            retrieval_without_gold += 1

        if result and result["predicted_handoff"]:
            predicted_handoffs += 1
            if case["task_kind"] == "handoff":
                correct_handoffs += 1

        if result and "generated_reply" in result:
            generated_replies += 1
            if result["response_locale"] == case["locale"]:
                locale_matches += 1
            if case["task_kind"] == "retrieval_answer":
                support_label = result.get("answer_support_label")
                if support_label is None:
                    answer_support_unannotated += 1
                else:
                    annotated_answers += 1
                    grounded_answers += support_label == "supported"
                    unsupported_answers += support_label == "unsupported"
        else:
            without_generated_reply += 1

    recall = _rate(relevant_retrieved, relevant_total)
    recall["k"] = k
    return {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "counts": {"cases": len(case_rows), "results": len(result_rows)},
        "metrics": {
            "recall_at_k": recall,
            "mrr": _rate(reciprocal_rank_total, retrieval_case_count),
            "grounded_answer_rate": _rate(grounded_answers, annotated_answers),
            "unsupported_answer_rate": _rate(unsupported_answers, annotated_answers),
            "handoff_precision": _rate(correct_handoffs, predicted_handoffs),
            "locale_consistency": _rate(locale_matches, generated_replies),
        },
        "excluded_cases": {
            "retrieval_without_gold": retrieval_without_gold,
            "answer_support_unannotated": answer_support_unannotated,
            "without_generated_reply": without_generated_reply,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate(
        load_jsonl(args.cases, expected_record_type="case"),
        load_jsonl(args.results, expected_record_type="result"),
        k=args.k,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
