import hashlib
from pathlib import Path

import pytest

from ml4qem_reuse.fetcher import Artifact, FetchError, acquire, artifacts_for_profile


def test_local_acquisition_resumes_and_verifies(tmp_path: Path) -> None:
    payload = b"verified release data" * 100
    source = tmp_path / "source.tar.gz"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    artifact = Artifact(
        identifier="test",
        filename="target.tar.gz",
        url=str(source),
        sha256=digest,
        bytes=len(payload),
        profiles=("analysis",),
    )
    partial = tmp_path / "cache/downloads/target.tar.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:37])

    acquired = acquire(artifact, tmp_path / "cache")

    assert acquired.read_bytes() == payload
    assert not partial.exists()


def test_cached_hash_mismatch_fails_without_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected")
    artifact = Artifact(
        identifier="test",
        filename="target.bin",
        url=str(source),
        sha256=hashlib.sha256(b"expected").hexdigest(),
        bytes=len(b"expected"),
        profiles=("analysis",),
    )
    cached = tmp_path / "cache/downloads/target.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"tampered")

    with pytest.raises(FetchError, match="SHA-256 mismatch"):
        acquire(artifact, tmp_path / "cache")

    assert cached.read_bytes() == b"tampered"


def test_registered_profiles_have_no_quantum_credentials() -> None:
    for profile in ("analysis", "benchmark", "full"):
        records = artifacts_for_profile(profile)
        assert records
        text = " ".join((record.url or "") + record.filename for record in records).lower()
        assert "token" not in text
        assert "credential" not in text

