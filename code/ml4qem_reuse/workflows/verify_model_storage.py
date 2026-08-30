#!/usr/bin/env python3
"""Verify an omitted-model audit directory against its compact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path, manifest: Path, expected_bytes: int) -> dict[str, object]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_names = {row["filename"] for row in rows}
    actual_names = {path.name for path in model_dir.glob("*.joblib") if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"model file set mismatch: missing={missing}, extra={extra}")

    total = 0
    for row in rows:
        path = model_dir / row["filename"]
        size = path.stat().st_size
        total += size
        if size != int(row["logical_bytes"]):
            raise ValueError(f"size mismatch for {path.name}: {size}")
        digest = sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path.name}: {digest}")
    if total != expected_bytes:
        raise ValueError(f"aggregate byte mismatch: {total} != {expected_bytes}")
    return {
        "file_count": len(rows),
        "serialized_bytes": total,
        "manifest_sha256": sha256(manifest),
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_dir, args.manifest, args.expected_bytes), indent=2))


if __name__ == "__main__":
    main()
