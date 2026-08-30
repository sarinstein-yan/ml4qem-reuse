"""Paper-era recovery used by the opt-in ``full`` profile."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml4qem_reuse.fetcher import FetchError, default_cache_dir, locate_data_root


LEGACY_ROOT_NAME = "qiskit-community-ml-qem-9776e1b"
SOURCE_DATA_NAME = (
    "Liao et al. - 2024 - Machine learning for practical quantum error mitigation "
    "- Source data.xlsx"
)


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "patches/legacy-rb-field.patch").is_file():
        raise FetchError(
            "The full profile needs the source checkout containing patches/ and environments/."
        )
    return root


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    display = " ".join(command)
    print(f"[full] {display}", flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        raise FetchError(f"Full-profile command failed ({completed.returncode}): {display}")


def _apply_patch(snapshot: Path, patch_path: Path) -> str:
    target = snapshot / "blackwater/data/generators/rb.py"
    text = target.read_text(encoding="utf-8")
    if "noisy_exp_values=[noisy_exp_val]" in text:
        return "already_applied"
    _run(["patch", "-p1", "--batch", "--forward", "-i", str(patch_path)], cwd=snapshot)
    if "noisy_exp_values=[noisy_exp_val]" not in target.read_text(encoding="utf-8"):
        raise FetchError("Legacy one-line repair did not produce the expected source line")
    return "applied"


def _compare_npz(left: Path, right: Path) -> dict[str, Any]:
    with np.load(left, allow_pickle=False) as first, np.load(right, allow_pickle=False) as second:
        first_names = set(first.files)
        second_names = set(second.files)
        if first_names != second_names:
            raise FetchError(
                f"NPZ members differ: {sorted(first_names - second_names)} / "
                f"{sorted(second_names - first_names)}"
            )
        differences = {}
        for name in sorted(first_names):
            if first[name].shape != second[name].shape:
                raise FetchError(f"NPZ shape differs for {name}: {first[name].shape} vs {second[name].shape}")
            if np.issubdtype(first[name].dtype, np.number):
                maximum = float(np.max(np.abs(first[name].astype(float) - second[name].astype(float))))
                differences[name] = maximum
            elif not np.array_equal(first[name], second[name]):
                raise FetchError(f"NPZ nonnumeric member differs: {name}")
        global_maximum = max(differences.values(), default=0.0)
        if global_maximum != 0.0:
            raise FetchError(f"Legacy stable export differs from the distributed extract: {global_maximum}")
    return {"members": len(first_names), "global_max_abs_difference": global_maximum}


def reproduce_full(output_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """Rebuild the legacy environment, stable exports, and portability checks."""

    cache = (cache_dir or default_cache_dir()).resolve()
    data_root = locate_data_root(cache)
    immutable_snapshot = cache / "legacy" / LEGACY_ROOT_NAME
    if not immutable_snapshot.is_dir():
        raise FetchError(
            f"Legacy snapshot is absent at {immutable_snapshot}. "
            "Run 'ml4qem-reuse fetch --profile full' first."
        )
    publisher_source = cache / "downloads" / "ml4qem-publisher-source-data.xlsx"
    if not publisher_source.is_file():
        raise FetchError(f"Publisher Source Data is absent from the cache: {publisher_source}")

    repository = _repository_root()
    output = output_dir.resolve()
    workspace = output / "workspace"
    snapshot = workspace / "upstream/snapshots" / LEGACY_ROOT_NAME
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(immutable_snapshot, snapshot)
    repair_status = _apply_patch(snapshot, repository / "patches/legacy-rb-field.patch")

    (workspace / "code").mkdir(parents=True, exist_ok=True)
    (workspace / "environments").mkdir(parents=True, exist_ok=True)
    (workspace / "refs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository / "code/bootstrap_legacy_environment.sh", workspace / "code")
    shutil.copy2(
        repository / "environments/legacy-pip-freeze.txt",
        workspace / "environments/legacy-pip-freeze.txt",
    )
    shutil.copy2(publisher_source, workspace / "refs" / SOURCE_DATA_NAME)

    environment_dir = workspace / ".venv-legacy"
    if not environment_dir.exists():
        bootstrap_env = os.environ.copy()
        bootstrap_env["LEGACY_ENVIRONMENT_DIR"] = str(environment_dir)
        _run(["bash", "code/bootstrap_legacy_environment.sh"], cwd=workspace, env=bootstrap_env)

    legacy_python = environment_dir / "bin/python"
    if not legacy_python.is_file():
        raise FetchError(f"Legacy interpreter was not created: {legacy_python}")

    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(repository / "code")
    run_env["ML4QEM_REUSE_WORKSPACE"] = str(workspace)
    run_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    checks = output / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    _run([str(legacy_python), "-m", "pytest", "-q", "tests"], cwd=snapshot, env=run_env)
    _run(
        [
            str(legacy_python),
            "-m",
            "ml4qem_reuse.workflows.audit_published_results",
            "--output",
            str(checks / "published_result_audit.json"),
        ],
        cwd=repository,
        env=run_env,
    )
    _run(
        [
            str(legacy_python),
            "-m",
            "ml4qem_reuse.workflows.export_published_predictions",
            "--output",
            str(checks / "published_predictions.npz"),
            "--manifest",
            str(checks / "published_predictions_manifest.json"),
        ],
        cwd=repository,
        env=run_env,
    )
    export_comparison = _compare_npz(
        checks / "published_predictions.npz",
        data_root / "data/derived/published_predictions.npz",
    )

    _run(
        [
            sys.executable,
            "-m",
            "ml4qem_reuse.workflows.check_feature_equivalence",
            "--input",
            str(data_root / "data/derived/legacy_equivalence_cases.jsonl.gz"),
            "--output",
            str(checks / "feature_equivalence.json"),
            "--atol",
            "5e-7",
        ],
        cwd=repository,
        env=run_env,
    )

    portability = {}
    for demo in ("demo1", "demo2"):
        legacy_predictions = checks / f"{demo}_30seeds_predictions.npz"
        _run(
            [
                str(legacy_python),
                "-m",
                "ml4qem_reuse.workflows.reproduce_demos",
                "--demo",
                demo,
                "--seeds",
                "30",
                "--trees",
                "100",
                "--n-jobs",
                "1",
                "--output",
                str(checks / f"{demo}_30seeds.json"),
                "--predictions",
                str(legacy_predictions),
                "--design",
                str(data_root / f"data/derived/{demo}_design.npz"),
            ],
            cwd=repository,
            env=run_env,
        )
        portability_json = checks / f"{demo}_rf_portability.json"
        _run(
            [
                sys.executable,
                "-m",
                "ml4qem_reuse.workflows.check_rf_portability",
                "--design",
                str(data_root / f"data/derived/{demo}_design.npz"),
                "--legacy-predictions",
                str(legacy_predictions),
                "--trees",
                "100",
                "--n-jobs",
                "1",
                "--output",
                str(portability_json),
                "--predictions",
                str(checks / f"{demo}_rf_portability_predictions.npz"),
            ],
            cwd=repository,
            env=run_env,
        )
        portability[demo] = json.loads(portability_json.read_text(encoding="utf-8"))[
            "global_prediction_max_absolute_difference"
        ]

    result = {
        "schema_version": 1,
        "profile": "full",
        "legacy_archive_root": LEGACY_ROOT_NAME,
        "legacy_repair": repair_status,
        "legacy_tests": "passed",
        "stable_export": export_comparison,
        "feature_equivalence": "passed_at_5e-7",
        "rf_portability_max_abs_difference": portability,
        "hardware_jobs_submitted": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result

