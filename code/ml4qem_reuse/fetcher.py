"""Verified, resumable acquisition of the public data dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class FetchError(RuntimeError):
    """Raised when an artifact cannot be acquired and verified safely."""


@dataclass(frozen=True)
class Artifact:
    """One remotely or locally acquired immutable artifact."""

    identifier: str
    filename: str
    url: str | None
    sha256: str
    bytes: int
    profiles: tuple[str, ...]
    archive: str | None = None
    archive_root: str | None = None
    environment_url: str | None = None


def default_cache_dir() -> Path:
    configured = os.environ.get("ML4QEM_REUSE_CACHE")
    return Path(configured or ".cache/ml4qem-reuse").expanduser().resolve()


def _registry_path() -> Path:
    source_registry = Path(__file__).resolve().parents[2] / "DATA_REGISTRY.yaml"
    if source_registry.is_file():
        return source_registry
    packaged = resources.files("ml4qem_reuse").joinpath("registry.yaml")
    return Path(str(packaged))


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = (path or _registry_path()).resolve()
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FetchError(f"Cannot read data registry {registry_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise FetchError(f"Malformed data registry: {registry_path}")
    return payload


def _artifact(record: dict[str, Any]) -> Artifact:
    required = ("id", "filename", "sha256", "bytes", "profiles")
    missing = [field for field in required if field not in record]
    if missing:
        raise FetchError(f"Registry artifact is missing fields {missing}: {record!r}")
    return Artifact(
        identifier=str(record["id"]),
        filename=str(record["filename"]),
        url=None if record.get("url") in (None, "") else str(record["url"]),
        sha256=str(record["sha256"]),
        bytes=int(record["bytes"]),
        profiles=tuple(str(value) for value in record["profiles"]),
        archive=None if record.get("archive") in (None, "") else str(record["archive"]),
        archive_root=(
            None if record.get("archive_root") in (None, "") else str(record["archive_root"])
        ),
        environment_url=(
            None
            if record.get("environment_url") in (None, "")
            else str(record["environment_url"])
        ),
    )


def artifacts_for_profile(profile: str, registry: dict[str, Any] | None = None) -> list[Artifact]:
    if profile not in {"analysis", "benchmark", "full"}:
        raise FetchError(f"Unknown profile {profile!r}; choose analysis, benchmark, or full")
    payload = registry or load_registry()
    return [
        artifact
        for artifact in (_artifact(record) for record in payload["artifacts"])
        if profile in artifact.profiles
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, *, expected_sha256: str, expected_bytes: int) -> None:
    if not path.is_file():
        raise FetchError(f"Required file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise FetchError(
            f"Byte-count mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FetchError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def _resolve_url(artifact: Artifact) -> str:
    if artifact.identifier == "ml4qem_reuse_data_archive":
        override = os.environ.get("ML4QEM_REUSE_DATA_URL")
        if override:
            return override
    if artifact.url:
        return artifact.url
    variable = artifact.environment_url or "ML4QEM_REUSE_DATA_URL"
    raise FetchError(
        f"No URL is registered for {artifact.identifier}. Set {variable} to a local "
        "archive path, file:// URL, or trusted mirror."
    )


def _local_path(url: str) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path)).expanduser().resolve()
    if parsed.scheme == "":
        candidate = Path(url).expanduser()
        if candidate.exists() or candidate.is_absolute():
            return candidate.resolve()
    return None


def _copy_local_resumable(source: Path, partial: Path) -> None:
    if not source.is_file():
        raise FetchError(f"Local artifact mirror does not exist: {source}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source.stat().st_size:
        raise FetchError(f"Partial download is larger than source: {partial}")
    mode = "ab" if offset else "wb"
    with source.open("rb") as source_stream, partial.open(mode) as target_stream:
        source_stream.seek(offset)
        shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)


def _download_http_resumable(url: str, partial: Path, *, retries: int = 4) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "ml4qem-reuse/1.0.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise FetchError(f"Download failed after {retries} attempts for {url}: {last_error}")


def acquire(artifact: Artifact, cache_dir: Path) -> Path:
    downloads = cache_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    destination = downloads / artifact.filename
    if destination.exists():
        verify_file(
            destination,
            expected_sha256=artifact.sha256,
            expected_bytes=artifact.bytes,
        )
        return destination

    url = _resolve_url(artifact)
    partial = destination.with_name(destination.name + ".part")
    local_source = _local_path(url)
    if local_source is not None:
        _copy_local_resumable(local_source, partial)
    else:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise FetchError(f"Unsupported artifact URL scheme for {artifact.identifier}: {url}")
        _download_http_resumable(url, partial)
    verify_file(partial, expected_sha256=artifact.sha256, expected_bytes=artifact.bytes)
    os.replace(partial, destination)
    return destination


def _safe_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _verify_manifest(root: Path) -> int:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FetchError(f"Extracted data archive has no MANIFEST.sha256: {root}")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise FetchError(f"Malformed manifest line in {manifest}: {line!r}") from exc
        target = root / relative
        if not _safe_member(relative):
            raise FetchError(f"Unsafe path in {manifest}: {relative}")
        verify_file(target, expected_sha256=digest, expected_bytes=target.stat().st_size)
        checked += 1
    return checked


def extract(artifact: Artifact, archive_path: Path, cache_dir: Path) -> Path | None:
    if artifact.archive is None:
        return None
    if artifact.archive_root is None:
        raise FetchError(f"Archive root is missing for {artifact.identifier}")
    category = "datasets" if artifact.identifier == "ml4qem_reuse_data_archive" else "legacy"
    destination_parent = cache_dir / category
    root = destination_parent / artifact.archive_root
    marker = root / ".ml4qem_reuse_extracted.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("archive_sha256") != artifact.sha256:
            raise FetchError(f"Extraction marker does not match registered archive: {marker}")
        if artifact.identifier == "ml4qem_reuse_data_archive":
            _verify_manifest(root)
        return root

    destination_parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_parent / f".{artifact.archive_root}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        if artifact.archive == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as archive:
                unsafe = [member.name for member in archive.getmembers() if not _safe_member(member.name)]
                if unsafe:
                    raise FetchError(f"Unsafe members in {archive_path}: {unsafe[:3]}")
                archive.extractall(temporary, filter="data")
        elif artifact.archive == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                unsafe = [name for name in archive.namelist() if not _safe_member(name)]
                if unsafe:
                    raise FetchError(f"Unsafe members in {archive_path}: {unsafe[:3]}")
                archive.extractall(temporary)
        else:
            raise FetchError(f"Unsupported archive type {artifact.archive!r}")
        extracted = temporary / artifact.archive_root
        if not extracted.is_dir():
            raise FetchError(
                f"Archive {archive_path} did not contain expected root {artifact.archive_root}"
            )
        if artifact.identifier == "ml4qem_reuse_data_archive":
            _verify_manifest(extracted)
        marker_payload = {
            "archive_sha256": artifact.sha256,
            "archive_bytes": artifact.bytes,
            "artifact_id": artifact.identifier,
        }
        (extracted / ".ml4qem_reuse_extracted.json").write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(extracted, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root


def fetch_profile(profile: str, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    cache = (cache_dir or default_cache_dir()).resolve()
    records = []
    for artifact in artifacts_for_profile(profile):
        path = acquire(artifact, cache)
        extracted = extract(artifact, path, cache)
        records.append(
            {
                "id": artifact.identifier,
                "path": str(path),
                "sha256": artifact.sha256,
                "bytes": artifact.bytes,
                "extracted_root": None if extracted is None else str(extracted),
            }
        )
    return records


def locate_data_root(cache_dir: Path | None = None) -> Path:
    cache = (cache_dir or default_cache_dir()).resolve()
    root = cache / "datasets" / "ml4qem-reuse-data-v1.0.0"
    if not root.is_dir():
        raise FetchError(
            f"Fetched data are absent at {root}. Run 'ml4qem-reuse fetch --profile analysis' first."
        )
    _verify_manifest(root)
    return root
