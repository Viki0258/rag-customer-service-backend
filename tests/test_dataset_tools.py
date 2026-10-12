from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import zipfile

import pytest


SCHEMA_VERSION = "1.0"


def _optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


download_public = _optional_import("scripts.datasets.download_public")
convert_public = _optional_import("scripts.datasets.convert_public")
run_eval = _optional_import("scripts.eval.run_eval")


def _require(module: Any | None, name: str) -> Any:
    assert module is not None, f"Task 8 module {name} must exist"
    return module


def _case(
    case_id: str,
    *,
    locale: str = "en",
    task_kind: str = "retrieval_answer",
    relevant: list[str] | None = None,
    support_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    relevant_ids = set(relevant or [])
    row: dict[str, Any] = {
        "record_type": "case",
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "locale": locale,
        "question": f"Question {case_id}",
        "task_kind": task_kind,
        "evidence_candidates": {
            "e1": {"text": "first", "expected_relevant": "e1" in relevant_ids},
            "e2": {"text": "second", "expected_relevant": "e2" in relevant_ids},
        },
        "source_dataset": "project-synthetic",
        "synthetic": True,
    }
    if support_labels is not None:
        row["support_labels"] = support_labels
    return row


def _result(
    case_id: str,
    *,
    retrieved: list[str] | None = None,
    handoff: bool = False,
    reply: str | None = None,
    response_locale: str | None = None,
    answer_support_label: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": "result",
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "retrieved_evidence_ids": retrieved or [],
        "predicted_handoff": handoff,
    }
    if reply is not None:
        row["generated_reply"] = reply
    if response_locale is not None:
        row["response_locale"] = response_locale
    if answer_support_label is not None:
        row["answer_support_label"] = answer_support_label
    return row


@pytest.mark.parametrize(
    "relative",
    [
        Path(".rag-data") / "dataset.jsonl",
        Path(".rag-data") / "nested" / "dataset.jsonl",
    ],
)
def test_destination_accepts_only_files_inside_resolved_rag_data(
    tmp_path: Path, relative: Path
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()

    resolved = module.resolve_destination(repository / relative, repository)

    assert resolved == (repository / relative).resolve()


@pytest.mark.parametrize(
    "destination",
    [
        Path("..") / "escape.jsonl",
        Path(".rag-data-evil") / "prefix-lookalike.jsonl",
    ],
)
def test_destination_rejects_traversal_and_prefix_lookalikes(
    tmp_path: Path, destination: Path
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(ValueError, match=r"\.rag-data"):
        module.resolve_destination(repository / destination, repository)


def test_destination_rejects_absolute_sibling(tmp_path: Path) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(ValueError, match=r"\.rag-data"):
        module.resolve_destination(tmp_path / "sibling" / "data.jsonl", repository)


def test_destination_rejects_symlink_that_resolves_outside_rag_data(
    tmp_path: Path,
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    data_root = repository / ".rag-data"
    outside = tmp_path / "outside"
    data_root.mkdir(parents=True)
    outside.mkdir()
    link = data_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match=r"\.rag-data"):
        module.resolve_destination(link / "dataset.jsonl", repository)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_destination_rejects_junction_that_resolves_outside_rag_data(
    tmp_path: Path,
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    data_root = repository / ".rag-data"
    outside = tmp_path / "outside"
    data_root.mkdir(parents=True)
    outside.mkdir()
    junction = data_root / "escape-junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junctions unavailable: {created.stderr or created.stdout}")
    try:
        with pytest.raises(ValueError, match=r"\.rag-data"):
            module.resolve_destination(junction / "dataset.jsonl", repository)
    finally:
        junction.rmdir()


class _FailingStream:
    def __enter__(self) -> _FailingStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        raise OSError("connection interrupted")


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, effective_url: str) -> None:
        super().__init__(payload)
        self._effective_url = effective_url
        self.headers = {"Content-Disposition": 'attachment; filename="evil.exe"'}

    def geturl(self) -> str:
        return self._effective_url


class _ZipMetadata:
    def __init__(self, members: list[zipfile.ZipInfo]) -> None:
        self._members = members

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self._members


def _zip_info(
    name: str,
    *,
    size: int = 1,
    compressed_size: int = 1,
    mode: int = 0,
    flags: int = 0,
) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(name)
    member.file_size = size
    member.compress_size = compressed_size
    member.external_attr = mode << 16
    member.flag_bits = flags
    return member


def _write_multidoc_archive(path: Path) -> None:
    documents = {
        "doc_data": {
            "dmv": {
                "Returns page": {
                    "doc_id": "doc-returns",
                    "title": "Returns",
                    "doc_text": "Returns are accepted for 30 days. Shipping takes 3 days.",
                    "spans": {
                        "1": {"id_sp": "1", "text_sp": "Returns are accepted for 30 days."},
                        "2": {"id_sp": "2", "text_sp": "Shipping takes 3 days."},
                    },
                },
                "Accounts page": {
                    "doc_id": "doc-account",
                    "title": "Accounts",
                    "doc_text": "Bring identification to update an account.",
                    "spans": {
                        "9": {"id_sp": "9", "text_sp": "Bring identification."},
                    },
                },
            }
        }
    }
    dialogues = {
        "dial_data": {
            "dmv": [
                {
                    "dial_id": "dialogue-1",
                    "turns": [
                        {
                            "turn_id": 1,
                            "role": "user",
                            "utterance": "How long is the return window?",
                        },
                        {
                            "turn_id": 2,
                            "role": "agent",
                            "utterance": "Thirty days.",
                            "references": [{"doc_id": "doc-returns", "id_sp": "1"}],
                        },
                        {
                            "turn_id": 3,
                            "role": "user",
                            "utterance": "What else should I know?",
                        },
                        {
                            "turn_id": 4,
                            "role": "agent",
                            "utterance": "Shipping and account details differ.",
                            "references": [
                                {"doc_id": "doc-returns", "id_sp": "2"},
                                {"doc_id": "doc-account", "id_sp": "9"},
                            ],
                        },
                    ],
                }
            ]
        }
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("multidoc2dial/multidoc2dial_doc.json", json.dumps(documents))
        archive.writestr(
            "multidoc2dial/multidoc2dial_dial_train.json", json.dumps(dialogues)
        )


def test_download_is_atomic_and_records_provenance(tmp_path: Path) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "emanual.parquet"
    payload = b"bounded fake parquet bytes"
    seen_urls: list[str] = []

    def fetch(url: str) -> _Response:
        seen_urls.append(url)
        return _Response(payload, "https://cdn-lfs.huggingface.co/fake/emanual.parquet")

    metadata_path = module.download_dataset(
        "ragbench-emanual",
        destination,
        repository_root=repository,
        fetch=fetch,
        retrieved_at="2026-09-06T12:00:00Z",
    )

    assert destination.read_bytes() == payload
    assert metadata_path == destination.with_name("emanual.parquet.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "checksum": {"algorithm": "sha256", "value": hashlib.sha256(payload).hexdigest()},
        "converter_version": "1.0",
        "dataset": "ragbench-emanual",
        "download_url": "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/emanual/test-00000-of-00001.parquet",
        "effective_url": "https://cdn-lfs.huggingface.co/fake/emanual.parquet",
        "license": "CC-BY-4.0",
        "retrieved_at": "2026-09-06T12:00:00Z",
        "revision": "97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
        "schema_version": "1.0",
        "selected_config": "emanual",
        "selected_split": "test",
        "source_url": "https://huggingface.co/datasets/galileo-ai/ragbench/raw/main/README.md",
    }
    assert seen_urls == [
        "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/emanual/test-00000-of-00001.parquet"
    ]
    assert not (destination.parent / "evil.exe").exists()
    assert not list(destination.parent.glob("*.part"))


def test_download_failure_leaves_no_final_or_partial_file(tmp_path: Path) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "broken.json"

    with pytest.raises(OSError, match="interrupted"):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=lambda _url: _FailingStream(),
        )

    assert not destination.exists()
    assert not destination.with_name("broken.json.metadata.json").exists()
    assert not list(destination.parent.glob("*.part"))


def test_download_rejects_oversized_response_without_publishing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "dataset.zip"
    monkeypatch.setattr(module, "MAX_DOWNLOAD_BYTES", 8)

    with pytest.raises(ValueError, match="download exceeds"):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=lambda _url: _Response(
                b"123456789", "https://example.invalid/oversized"
            ),
        )

    assert not destination.exists()
    assert not destination.with_name("dataset.zip.metadata.json").exists()
    assert not list(destination.parent.glob("*.part"))


def test_existing_download_is_reused_only_when_recorded_checksum_matches(
    tmp_path: Path,
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "bitext.csv"
    payload = b"fake,csv\n"
    module.download_dataset(
        "bitext-customer-support",
        destination,
        repository_root=repository,
        fetch=lambda _url: io.BytesIO(payload),
        retrieved_at="2026-09-06T12:00:00Z",
    )

    reused = module.download_dataset(
        "bitext-customer-support",
        destination,
        repository_root=repository,
        fetch=lambda _url: pytest.fail("matching existing file must not redownload"),
    )
    assert reused == destination.with_name("bitext.csv.metadata.json")

    destination.write_bytes(b"tampered")
    with pytest.raises(FileExistsError, match="checksum"):
        module.download_dataset(
            "bitext-customer-support",
            destination,
            repository_root=repository,
            fetch=lambda _url: io.BytesIO(payload),
        )


def test_existing_download_rejects_cross_dataset_identity(tmp_path: Path) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "shared.bin"
    module.download_dataset(
        "multidoc2dial",
        destination,
        repository_root=repository,
        fetch=lambda _url: _Response(b"same bytes", "https://doc2dial.github.io/file.zip"),
    )

    with pytest.raises(FileExistsError, match="identity"):
        module.download_dataset(
            "bitext-customer-support",
            destination,
            repository_root=repository,
            fetch=lambda _url: pytest.fail("identity mismatch must not redownload"),
        )


@pytest.mark.parametrize("failed_replace", [1, 2])
def test_publish_failure_never_leaves_an_inconsistent_final_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_replace: int
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "dataset.zip"
    metadata_path = destination.with_name("dataset.zip.metadata.json")
    real_replace = module.os.replace
    calls = 0

    def fail_at_boundary(source: Path, target: Path) -> None:
        nonlocal calls
        if Path(target) not in {destination, metadata_path}:
            real_replace(source, target)
            return
        calls += 1
        if calls == failed_replace:
            raise OSError(f"replace {failed_replace} failed")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_at_boundary)
    with pytest.raises(OSError, match=rf"replace {failed_replace} failed"):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=lambda _url: _Response(b"archive", "https://example.invalid/effective"),
        )

    assert not destination.exists()
    assert not metadata_path.exists()
    assert not list(destination.parent.glob("*.part"))


@pytest.mark.parametrize(
    ("transition", "expected_fetches"),
    [
        ("journal_prepared", 2),
        ("metadata_published", 2),
        ("data_published", 1),
    ],
)
def test_next_download_recovers_process_death_after_each_publish_transition(
    tmp_path: Path, transition: str, expected_fetches: int
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "dataset.zip"
    fetches = 0

    def fetch(_url: str) -> _Response:
        nonlocal fetches
        fetches += 1
        return _Response(b"archive", "https://example.invalid/effective")

    def terminate(stage: str) -> None:
        if stage == transition:
            raise SystemExit("simulated process death")

    with pytest.raises(SystemExit, match="simulated process death"):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=fetch,
            retrieved_at="2026-09-07T00:00:00Z",
            _after_publish_transition=terminate,
        )

    metadata_path = module.download_dataset(
        "multidoc2dial",
        destination,
        repository_root=repository,
        fetch=fetch,
        retrieved_at="2026-09-07T00:00:00Z",
    )

    assert fetches == expected_fetches
    assert destination.read_bytes() == b"archive"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["checksum"][
        "value"
    ] == hashlib.sha256(b"archive").hexdigest()
    assert not list(destination.parent.glob("*.publish-transaction.json"))


def test_recovery_never_deletes_final_file_that_does_not_match_journal(
    tmp_path: Path,
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    repository.mkdir()
    destination = repository / ".rag-data" / "dataset.zip"

    def terminate_after_data(stage: str) -> None:
        if stage == "data_published":
            raise SystemExit("simulated process death")

    with pytest.raises(SystemExit):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=lambda _url: _Response(
                b"archive", "https://example.invalid/effective"
            ),
            _after_publish_transition=terminate_after_data,
        )
    destination.write_bytes(b"unrelated user file")

    with pytest.raises(FileExistsError, match="transaction"):
        module.download_dataset(
            "multidoc2dial",
            destination,
            repository_root=repository,
            fetch=lambda _url: pytest.fail("unsafe recovery must stop before fetch"),
        )

    assert destination.read_bytes() == b"unrelated user file"


def test_only_three_public_dataset_names_are_supported() -> None:
    module = _require(download_public, "download_public")
    assert set(module.DATASETS) == {
        "multidoc2dial",
        "ragbench-emanual",
        "bitext-customer-support",
    }
    assert module.DATASETS["multidoc2dial"].license == "Apache-2.0"
    assert module.DATASETS["ragbench-emanual"].license == "CC-BY-4.0"
    assert module.DATASETS["bitext-customer-support"].license == "CDLA-Sharing-1.0"
    assert module.DATASETS["multidoc2dial"].revision is None
    assert module.DATASETS["ragbench-emanual"].revision == (
        "97808f3e5fd16ede40bbff6c2949af8139b2eb7b"
    )
    assert module.DATASETS["bitext-customer-support"].revision == (
        "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
    )


@pytest.mark.parametrize(
    ("dataset", "source_url", "download_url"),
    [
        (
            "multidoc2dial",
            "https://github.com/IBM/multidoc2dial",
            "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip",
        ),
        (
            "ragbench-emanual",
            "https://huggingface.co/datasets/galileo-ai/ragbench/raw/main/README.md",
            "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/emanual/test-00000-of-00001.parquet",
        ),
        (
            "bitext-customer-support",
            "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/blob/main/README.md",
            "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/430d1a89bd93bd1fa23c16f29dd53e73f0087443/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
        ),
    ],
)
def test_public_dataset_urls_are_exact_publisher_endpoints(
    dataset: str, source_url: str, download_url: str
) -> None:
    source = _require(download_public, "download_public").DATASETS[dataset]
    assert source.source_url == source_url
    assert source.download_url == download_url


def test_download_cli_requires_explicit_network_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _require(download_public, "download_public")
    monkeypatch.setattr(
        module,
        "download_dataset",
        lambda *_args, **_kwargs: pytest.fail("network boundary must remain closed"),
    )

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "multidoc2dial",
                str(tmp_path / ".rag-data" / "dataset.zip"),
            ]
        )

    assert raised.value.code == 2


def test_repository_root_discovery_ignores_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _require(download_public, "download_public")
    monkeypatch.chdir(tmp_path)

    assert module.discover_repository_root() == Path(__file__).resolve().parents[1]


def test_download_cli_confines_relative_destination_to_discovered_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _require(download_public, "download_public")
    repository = tmp_path / "repo"
    unrelated = tmp_path / "unrelated"
    repository.mkdir()
    unrelated.mkdir()
    (repository / ".git").mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    monkeypatch.chdir(unrelated)

    assert (
        module.main(
            ["multidoc2dial", ".rag-data/download.zip", "--allow-network"],
            fetch=lambda _url: _Response(b"archive", "https://example.invalid/effective"),
            _repository_root=repository,
        )
        == 0
    )
    assert (repository / ".rag-data" / "download.zip").read_bytes() == b"archive"
    assert not (unrelated / ".rag-data").exists()

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "multidoc2dial",
                ".rag-data/download.zip",
                "--allow-network",
                "--repository-root",
                str(unrelated),
            ],
            fetch=lambda _url: pytest.fail("removed option must be rejected"),
            _repository_root=repository,
        )
    assert raised.value.code == 2


def test_multidoc2dial_archive_uses_authoritative_shape_and_grounded_spans(
    tmp_path: Path,
) -> None:
    module = _require(convert_public, "convert_public")
    archive = tmp_path / "multidoc2dial.zip"
    _write_multidoc_archive(archive)

    records = module.read_records(archive, dataset="multidoc2dial")
    converted = module.convert_records("multidoc2dial", records)

    assert [case["case_id"] for case in converted] == [
        "multidoc2dial:dialogue-1:2",
        "multidoc2dial:dialogue-1:4",
    ]
    assert converted[0]["question"] == "How long is the return window?"
    assert [
        evidence_id
        for evidence_id, candidate in converted[0]["evidence_candidates"].items()
        if candidate["expected_relevant"]
    ] == ["doc-returns:1"]
    assert list(converted[0]["evidence_candidates"]) == [
        "doc-account:9",
        "doc-returns:1",
        "doc-returns:2",
    ]
    assert converted[0]["source_dataset"] == "multidoc2dial"
    assert converted[0]["synthetic"] is False


@pytest.mark.parametrize(
    ("member_name", "payload_kind", "message"),
    [
        ("../escape.json", "small", "unsafe ZIP member"),
        ("oversized.json", "compressible", "compression ratio"),
    ],
)
def test_multidoc_archive_rejects_unsafe_members(
    tmp_path: Path, member_name: str, payload_kind: str, message: str
) -> None:
    module = _require(convert_public, "convert_public")
    archive_path = tmp_path / "unsafe.zip"
    payload = b"{}" if payload_kind == "small" else b"0" * (1024 * 1024)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, payload)

    with pytest.raises(ValueError, match=message):
        module.read_records(archive_path, dataset="multidoc2dial")


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([_zip_info("link.json", mode=0o120777)], "link"),
        ([_zip_info("encrypted.json", flags=0x1)], "encrypted"),
        ([_zip_info("large.json", size=129)], "size limit"),
        (
            [
                _zip_info("first.json", size=60),
                _zip_info("second.json", size=60),
            ],
            "total uncompressed size",
        ),
    ],
)
def test_multidoc_archive_rejects_remaining_unsafe_member_metadata(
    monkeypatch: pytest.MonkeyPatch,
    members: list[zipfile.ZipInfo],
    message: str,
) -> None:
    module = _require(convert_public, "convert_public")
    monkeypatch.setattr(module, "MAX_ZIP_MEMBER_BYTES", 128)
    monkeypatch.setattr(module, "MAX_ZIP_TOTAL_BYTES", 100)

    with pytest.raises(ValueError, match=message):
        module._validate_zip_members(_ZipMetadata(members))


def test_multidoc_archive_rejects_excess_member_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _require(convert_public, "convert_public")
    monkeypatch.setattr(module, "MAX_ZIP_MEMBERS", 2)

    with pytest.raises(ValueError, match="more than 2 members"):
        module._validate_zip_members(
            _ZipMetadata([_zip_info(f"{index}.json") for index in range(3)])
        )


def test_ragbench_retains_annotations_without_promoting_response_to_gold() -> None:
    module = _require(convert_public, "convert_public")
    records = [
        {
            "id": "emanual-7",
            "question": "Can I immerse the device?",
            "documents": ["Keep the device dry.", "Charge with the supplied cable."],
            "response": "It is waterproof.",
            "documents_sentences": [["Keep the device dry."], ["Charge with the supplied cable."]],
            "sentence_support_information": {"0": []},
            "adherence_score": 0.0,
            "unsupported_response_sentence_keys": ["0"],
            "all_relevant_sentence_keys": ["0"],
        }
    ]

    converted = module.convert_records("ragbench-emanual", records)

    case = converted[0]
    assert case["evidence_candidates"]["emanual-7:sentence:0"][
        "expected_relevant"
    ] is True
    assert case["support_labels"] == {"0": "unsupported"}
    assert case["source_annotations"] == {
        "adherence_score": 0.0,
        "all_relevant_sentence_keys": ["0"],
        "sentence_support_information": {"0": []},
        "unsupported_response_sentence_keys": ["0"],
    }
    assert "response" not in case
    assert "reference_answer" not in case


def test_ragbench_maps_document_sentence_keys_without_losing_support_links() -> None:
    module = _require(convert_public, "convert_public")
    converted = module.convert_records(
        "ragbench-emanual",
        [
            {
                "id": "emanual-keys",
                "question": "Which cable should I use?",
                "documents": ["Use cable A. Use cable B."],
                "response": "Use cable B.",
                "documents_sentences": [["Use cable A.", "Use cable B."]],
                "sentence_support_information": {"0": ["0b"]},
                "adherence_score": 1.0,
                "unsupported_response_sentence_keys": [],
                "all_relevant_sentence_keys": ["0b"],
            }
        ],
    )

    case = converted[0]
    assert list(case["evidence_candidates"]) == [
        "emanual-keys:sentence:0a",
        "emanual-keys:sentence:0b",
    ]
    assert case["evidence_candidates"]["emanual-keys:sentence:0b"][
        "expected_relevant"
    ] is True
    assert case["source_annotations"]["sentence_support_information"] == {"0": ["0b"]}


def test_bitext_is_routing_only_and_has_no_retrieval_gold() -> None:
    module = _require(convert_public, "convert_public")
    converted = module.convert_records(
        "bitext-customer-support",
        [
            {
                "instruction": "I need to cancel my order",
                "response": "I can help.",
                "intent": "cancel_order",
                "category": "ORDER",
                "tags": "cancel",
            }
        ],
    )

    assert converted == [
        {
            "record_type": "case",
            "schema_version": SCHEMA_VERSION,
            "case_id": "bitext-customer-support:b564d08d859d1046efd38d59b428951220cc60a4d3903bac654473a4c7cd014e",
            "locale": "en",
            "question": "I need to cancel my order",
            "task_kind": "routing_only",
            "evidence_candidates": {},
            "expected_intent": "cancel_order",
            "source_dataset": "bitext-customer-support",
            "synthetic": False,
        }
    ]


def test_conversion_is_deterministic_and_rejects_empty_or_unknown_records() -> None:
    module = _require(convert_public, "convert_public")
    first = {"instruction": "Track order", "intent": "track_order"}
    second = {"instruction": "I need to cancel my order", "intent": "cancel_order"}
    records = [first, second]

    converted = module.convert_records("bitext-customer-support", records)
    assert converted == module.convert_records("bitext-customer-support", list(reversed(records)))
    assert [row["case_id"] for row in converted] == [
        "bitext-customer-support:62035e4d315e9098fa490f4c26f9713f6d27ecab1521464d0a73d46c9229a88c",
        "bitext-customer-support:b564d08d859d1046efd38d59b428951220cc60a4d3903bac654473a4c7cd014e",
    ]
    with pytest.raises(ValueError, match="duplicate output case_id"):
        module.convert_records("bitext-customer-support", [first, first])
    with pytest.raises(ValueError, match="empty"):
        module.convert_records("bitext-customer-support", [{}])
    with pytest.raises(ValueError, match="unsupported dataset"):
        module.convert_records("other", records)


def test_parquet_without_optional_dependency_has_install_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _require(convert_public, "convert_public")
    real_import = __import__

    def rejecting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pyarrow"):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", rejecting_import)
    with pytest.raises(RuntimeError, match=r"pip install pyarrow"):
        module.read_records(tmp_path / "records.parquet")


def test_jsonl_validation_is_strict_and_duplicate_ids_fail(tmp_path: Path) -> None:
    module = _require(run_eval, "run_eval")
    valid = _case("case-1", relevant=["e1"])
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    assert module.load_jsonl(path, expected_record_type="case") == [valid]

    for bad_row, message in [
        ({**valid, "surprise": True}, "unknown fields"),
        ({**valid, "locale": "fr"}, "locale"),
        ({**valid, "task_kind": "chat"}, "task_kind"),
        ({**valid, "question": ""}, "question"),
    ]:
        path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            module.load_jsonl(path, expected_record_type="case")

    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        module.load_jsonl(path, expected_record_type="case")

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record_type"):
        module.load_jsonl(path, expected_record_type="case")

    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        module.load_jsonl(path, expected_record_type="case")

    path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        module.load_jsonl(path, expected_record_type="case")

    duplicate_evidence_key = (
        '{"record_type":"case","schema_version":"1.0","case_id":"duplicate-key",'
        '"locale":"en","question":"Question","task_kind":"retrieval_answer",'
        '"evidence_candidates":{"same":{"text":"first","expected_relevant":true},'
        '"same":{"text":"different","expected_relevant":false}},'
        '"source_dataset":"project-synthetic","synthetic":true}\n'
    )
    path.write_text(duplicate_evidence_key, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object key.*same"):
        module.load_jsonl(path, expected_record_type="case")


def test_result_schema_rejects_unknown_fields_and_duplicate_ids(tmp_path: Path) -> None:
    module = _require(run_eval, "run_eval")
    valid = _result("case-1")
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({**valid, "grounded": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        module.load_jsonl(path, expected_record_type="result")

    path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        module.load_jsonl(path, expected_record_type="result")


def test_runtime_and_json_schema_accept_and_reject_the_same_edge_cases() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    module = _require(run_eval, "run_eval")
    repository = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (repository / "docs" / "evaluation" / "rag-evaluation-schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.Draft202012Validator(schema)
    valid_case = _case("case", support_labels={"answer": "supported"})
    valid_result = _result("case", reply="Answer", response_locale="en")
    rows = [
        (valid_case, True),
        (valid_result, True),
        ({**valid_result, "generated_reply": None}, False),
        ({**_result("locale-only"), "response_locale": "en"}, False),
        ({**_result("support-only"), "answer_support_label": "supported"}, False),
        ({**valid_case, "support_labels": {"answer": "unknown"}}, False),
        ({**valid_case, "question": "   \t"}, False),
        ({**valid_case, "expected_relevant_evidence_ids": ["missing"]}, False),
        (
            {
                **valid_case,
                "evidence_candidates": [
                    {"evidence_id": "duplicate", "text": "first"},
                    {"evidence_id": "duplicate", "text": "different"},
                ],
            },
            False,
        ),
    ]

    for row, expected_valid in rows:
        runtime_valid = True
        try:
            if row.get("record_type") == "case":
                module.validate_case(row)
            else:
                module.validate_result(row)
        except ValueError:
            runtime_valid = False
        schema_valid = not list(validator.iter_errors(row))
        assert runtime_valid is expected_valid, row
        assert schema_valid is runtime_valid, row

def test_metrics_report_explicit_denominators_and_exclusions() -> None:
    module = _require(run_eval, "run_eval")
    cases = [
        _case("retrieval", relevant=["e1", "e2"]),
        _case("unannotated", locale="zh"),
        _case("handoff", task_kind="handoff"),
        _case("routing", task_kind="routing_only"),
        _case("injection", task_kind="prompt_injection"),
    ]
    results = [
        _result(
            "retrieval",
            retrieved=["noise", "e2"],
            reply="Grounded reply",
            response_locale="en",
            answer_support_label="supported",
        ),
        _result(
            "unannotated",
            reply="错误语言",
            response_locale="en",
        ),
        _result("handoff", handoff=True),
        _result("routing", handoff=True),
        _result("injection", retrieved=["e1"]),
    ]

    report = module.evaluate(cases, results, k=2)

    assert report == {
        "schema_version": SCHEMA_VERSION,
        "converter_version": "1.0",
        "counts": {"cases": 5, "results": 5},
        "metrics": {
            "recall_at_k": {"value": 0.5, "numerator": 1, "denominator": 2, "k": 2},
            "mrr": {"value": 0.5, "numerator": 0.5, "denominator": 1},
            "grounded_answer_rate": {"value": 1.0, "numerator": 1, "denominator": 1},
            "unsupported_answer_rate": {"value": 0.0, "numerator": 0, "denominator": 1},
            "handoff_precision": {"value": 0.5, "numerator": 1, "denominator": 2},
            "locale_consistency": {"value": 0.5, "numerator": 1, "denominator": 2},
        },
        "excluded_cases": {
            "retrieval_without_gold": 4,
            "answer_support_unannotated": 1,
            "without_generated_reply": 3,
        },
    }


def test_zero_denominator_metrics_are_zero_with_zero_counts() -> None:
    module = _require(run_eval, "run_eval")
    report = module.evaluate(
        [_case("routing", task_kind="routing_only")],
        [_result("routing")],
        k=5,
    )

    for metric in report["metrics"].values():
        assert metric["value"] == 0.0
        assert metric["denominator"] == 0


def test_citation_membership_does_not_create_grounding_success() -> None:
    module = _require(run_eval, "run_eval")
    case = _case("cited", relevant=["e1"])
    result = {
        **_result("cited", reply="Answer", response_locale="en"),
        "cited_evidence_ids": ["e1"],
    }

    report = module.evaluate([case], [result], k=1)

    assert report["metrics"]["grounded_answer_rate"] == {
        "value": 0.0,
        "numerator": 0,
        "denominator": 0,
    }
    assert report["excluded_cases"]["answer_support_unannotated"] == 1


def test_committed_schema_and_bilingual_fixtures_are_strict_and_synthetic() -> None:
    module = _require(run_eval, "run_eval")
    repository = Path(__file__).resolve().parents[1]
    schema_path = repository / "docs" / "evaluation" / "rag-evaluation-schema.json"
    assert schema_path.exists(), "Task 8 schema must exist"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$defs"]["case"]["additionalProperties"] is False
    assert schema["$defs"]["result"]["additionalProperties"] is False

    all_cases: list[dict[str, Any]] = []
    for locale in ("en", "zh"):
        fixture = repository / "fixtures" / "eval" / f"faq_{locale}.jsonl"
        rows = module.load_jsonl(fixture, expected_record_type="case")
        assert rows
        assert {row["locale"] for row in rows} == {locale}
        assert all(row["synthetic"] is True for row in rows)
        assert all(row["source_dataset"] == "project-synthetic" for row in rows)
        all_cases.extend(rows)

    assert {row["task_kind"] for row in all_cases} >= {
        "retrieval_answer",
        "handoff",
        "routing_only",
        "prompt_injection",
    }
    ids = [row["case_id"] for row in all_cases]
    assert len(ids) == len(set(ids))
