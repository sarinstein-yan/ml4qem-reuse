import csv
import hashlib

import pytest

from ml4qem_reuse.workflows.verify_model_storage import verify


def test_verify_model_storage_checks_files_sizes_and_hashes(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    payloads = {"fold0_rf.joblib": b"forest", "fold0_mlp.joblib": b"network"}
    rows = []
    for name, payload in payloads.items():
        (model_dir / name).write_bytes(payload)
        rows.append(
            {
                "filename": name,
                "logical_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "logical_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    result = verify(model_dir, manifest, sum(map(len, payloads.values())))
    assert result["status"] == "verified"
    assert result["file_count"] == 2

    (model_dir / "fold0_rf.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify(model_dir, manifest, sum(map(len, payloads.values())))
