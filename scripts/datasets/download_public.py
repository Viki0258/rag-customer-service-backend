"""Opt-in downloads of the three public datasets used by local evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable, ContextManager, Mapping
from urllib.request import Request, urlopen
import uuid


CONVERTER_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
PUBLISH_JOURNAL_VERSION = 1


@dataclass(frozen=True)
class DatasetSource:
    source_url: str
    download_url: str
    license: str
    revision: str | None
    selected_config: str
    selected_split: str


DATASETS = {
    "multidoc2dial": DatasetSource(
        source_url="https://github.com/IBM/multidoc2dial",
        download_url="https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip",
        license="Apache-2.0",
        revision=None,
        selected_config="multidoc2dial",
        selected_split="all-published",
    ),
    "ragbench-emanual": DatasetSource(
        source_url="https://huggingface.co/datasets/galileo-ai/ragbench/raw/main/README.md",
        download_url=(
            "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/"
            "97808f3e5fd16ede40bbff6c2949af8139b2eb7b/"
            "emanual/test-00000-of-00001.parquet"
        ),
        license="CC-BY-4.0",
        revision="97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
        selected_config="emanual",
        selected_split="test",
    ),
    "bitext-customer-support": DatasetSource(
        source_url=(
            "https://huggingface.co/datasets/bitext/"
            "Bitext-customer-support-llm-chatbot-training-dataset/blob/main/README.md"
        ),
        download_url=(
            "https://huggingface.co/datasets/bitext/"
            "Bitext-customer-support-llm-chatbot-training-dataset/resolve/"
            "430d1a89bd93bd1fa23c16f29dd53e73f0087443/"
            "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        ),
        license="CDLA-Sharing-1.0",
        revision="430d1a89bd93bd1fa23c16f29dd53e73f0087443",
        selected_config="default",
        selected_split="train",
    ),
}


Fetch = Callable[[str], ContextManager[BinaryIO]]
PublishTransitionHook = Callable[[str], None]


def resolve_destination(destination: Path | str, repository_root: Path | str) -> Path:
    """Resolve a dataset filename and prove it remains under this repo's .rag-data."""

    repository = Path(repository_root).resolve(strict=True)
    logical_data_root = repository / ".rag-data"
    resolved_data_root = logical_data_root.resolve(strict=False)
    if resolved_data_root != logical_data_root:
        raise ValueError("repository .rag-data must not be a symlink or junction")

    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve(strict=False)
    if resolved == resolved_data_root or not resolved.is_relative_to(resolved_data_root):
        raise ValueError("destination must be a file inside repository-root .rag-data")
    return resolved


def _metadata_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.metadata.json")


def _journal_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.publish-transaction.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
            newline="\n",
        ) as output:
            temp_path = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _recover_publication(destination: Path, metadata_path: Path) -> None:
    journal_path = _journal_path(destination)
    if not journal_path.exists():
        return
    if not journal_path.is_file():
        raise FileExistsError("publication transaction journal is not a file")
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FileExistsError("publication transaction journal is invalid") from exc
    required = {
        "version",
        "transaction_id",
        "data_name",
        "metadata_name",
        "data_sha256",
        "metadata_sha256",
    }
    if (
        not isinstance(journal, dict)
        or set(journal) != required
        or journal.get("version") != PUBLISH_JOURNAL_VERSION
        or journal.get("data_name") != destination.name
        or journal.get("metadata_name") != metadata_path.name
        or not _is_sha256(journal.get("transaction_id"))
        or not _is_sha256(journal.get("data_sha256"))
        or not _is_sha256(journal.get("metadata_sha256"))
    ):
        raise FileExistsError("publication transaction journal is invalid")

    expected = (
        (destination, journal["data_sha256"]),
        (metadata_path, journal["metadata_sha256"]),
    )
    for path, checksum in expected:
        if path.exists() and (not path.is_file() or _sha256(path) != checksum):
            raise FileExistsError(
                "publication transaction does not match existing final file"
            )
    if destination.exists() and metadata_path.exists():
        journal_path.unlink()
        return
    for path, _checksum in expected:
        path.unlink(missing_ok=True)
    journal_path.unlink()


def _authoritative_identity(dataset: str, source: DatasetSource) -> dict[str, str | None]:
    return {
        "converter_version": CONVERTER_VERSION,
        "dataset": dataset,
        "download_url": source.download_url,
        "license": source.license,
        "revision": source.revision,
        "schema_version": SCHEMA_VERSION,
        "selected_config": source.selected_config,
        "selected_split": source.selected_split,
        "source_url": source.source_url,
    }


def _matching_existing_download(
    destination: Path,
    metadata_path: Path,
    *,
    expected_identity: Mapping[str, str | None],
) -> bool:
    if not destination.exists():
        return False
    if not destination.is_file() or not metadata_path.is_file():
        raise FileExistsError("existing dataset has no usable recorded checksum")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity_matches = all(metadata.get(key) == value for key, value in expected_identity.items())
        effective_url = metadata["effective_url"]
        if not identity_matches or not isinstance(effective_url, str) or not effective_url:
            raise FileExistsError("existing dataset authoritative identity does not match request")
        recorded = metadata["checksum"]
        matches = recorded["algorithm"] == "sha256" and recorded["value"] == _sha256(destination)
    except FileExistsError:
        raise
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise FileExistsError("existing dataset has no usable recorded checksum") from exc
    if not matches:
        raise FileExistsError("existing dataset checksum does not match recorded checksum")
    return True


def _urlopen(url: str) -> ContextManager[BinaryIO]:
    request = Request(url, headers={"User-Agent": "whatsapp-rag-dataset-tool/1.0"})
    return urlopen(request, timeout=60)


def download_dataset(
    dataset: str,
    destination: Path | str,
    *,
    repository_root: Path | str,
    fetch: Fetch = _urlopen,
    retrieved_at: str | None = None,
    _after_publish_transition: PublishTransitionHook | None = None,
) -> Path:
    """Download one explicit dataset, atomically publishing data and provenance."""

    try:
        source = DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc

    final_path = resolve_destination(destination, repository_root)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after directory creation so Windows reparse points cannot hide in parents.
    final_path = resolve_destination(final_path, repository_root)
    metadata_path = _metadata_path(final_path)
    journal_path = _journal_path(final_path)
    _recover_publication(final_path, metadata_path)
    identity = _authoritative_identity(dataset, source)
    if _matching_existing_download(
        final_path,
        metadata_path,
        expected_identity=identity,
    ):
        return metadata_path
    if metadata_path.exists():
        raise FileExistsError("metadata exists without a checksum-matching dataset")

    data_temp: Path | None = None
    metadata_temp: Path | None = None
    effective_url = source.download_url
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{final_path.name}.",
            suffix=".part",
            dir=final_path.parent,
            delete=False,
        ) as output:
            data_temp = Path(output.name)
            digest = hashlib.sha256()
            downloaded_bytes = 0
            with fetch(source.download_url) as response:
                geturl = getattr(response, "geturl", None)
                if callable(geturl):
                    reported_url = geturl()
                    if isinstance(reported_url, str) and reported_url:
                        effective_url = reported_url
                while chunk := response.read(1024 * 1024):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"download exceeds {MAX_DOWNLOAD_BYTES} byte limit"
                        )
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())

        metadata = {
            **identity,
            "checksum": {"algorithm": "sha256", "value": digest.hexdigest()},
            "effective_url": effective_url,
            "retrieved_at": retrieved_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{metadata_path.name}.",
            suffix=".part",
            dir=final_path.parent,
            delete=False,
            newline="\n",
        ) as output:
            metadata_temp = Path(output.name)
            json.dump(metadata, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

        journal = {
            "version": PUBLISH_JOURNAL_VERSION,
            "transaction_id": uuid.uuid4().hex + uuid.uuid4().hex,
            "data_name": final_path.name,
            "metadata_name": metadata_path.name,
            "data_sha256": digest.hexdigest(),
            "metadata_sha256": _sha256(metadata_temp),
        }
        _write_json_atomic(journal_path, journal)
        if _after_publish_transition is not None:
            _after_publish_transition("journal_prepared")

        os.replace(metadata_temp, metadata_path)
        metadata_temp = None
        if _after_publish_transition is not None:
            _after_publish_transition("metadata_published")
        os.replace(data_temp, final_path)
        data_temp = None
        if _after_publish_transition is not None:
            _after_publish_transition("data_published")
        journal_path.unlink()
        return metadata_path
    except Exception:
        _recover_publication(final_path, metadata_path)
        raise
    finally:
        if data_temp is not None:
            data_temp.unlink(missing_ok=True)
        if metadata_temp is not None:
            metadata_temp.unlink(missing_ok=True)


def discover_repository_root(anchor: Path | None = None) -> Path:
    """Find the actual checkout from this script's location, never from cwd."""

    start = (anchor or Path(__file__)).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not discover repository root from dataset tool location")


def main(
    argv: list[str] | None = None,
    *,
    fetch: Fetch | None = None,
    _repository_root: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="explicitly permit the requested public dataset download",
    )
    args = parser.parse_args(argv)
    if not args.allow_network:
        parser.error("network is disabled; pass --allow-network to download")
    metadata_path = download_dataset(
        args.dataset,
        args.destination,
        repository_root=_repository_root or discover_repository_root(),
        fetch=fetch or _urlopen,
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
