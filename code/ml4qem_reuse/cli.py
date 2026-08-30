"""Command-line interface for verified acquisition and reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from ml4qem_reuse.fetcher import (
    FetchError,
    artifacts_for_profile,
    default_cache_dir,
    fetch_profile,
    locate_data_root,
    verify_file,
)


def _repository_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_root() -> Path | None:
    repository = _repository_root()
    if repository and (repository / "reference/frozen_summary.json").is_file():
        return repository / "reference"
    return None


def _verify_cached(profile: str, cache: Path) -> None:
    for artifact in artifacts_for_profile(profile):
        target = cache / "downloads" / artifact.filename
        if not target.is_file():
            raise FetchError(
                f"Required file is missing: {target}. "
                f"Run 'ml4qem-reuse fetch --profile {profile}' first."
            )
        verify_file(target, expected_sha256=artifact.sha256, expected_bytes=artifact.bytes)


def _run_workflow(module_name: str, arguments: list[str]) -> None:
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        module.main()
    finally:
        sys.argv = previous


def _analysis(output: Path, cache: Path) -> dict[str, Any]:
    _verify_cached("analysis", cache)
    data_root = locate_data_root(cache)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ML4QEM_REUSE_WORKSPACE"] = str(data_root)
    os.environ["ML4QEM_REUSE_OUTPUT_ROOT"] = str(output)
    _run_workflow("ml4qem_reuse.workflows.build_public_outputs", [])
    _run_workflow("ml4qem_reuse.workflows.make_public_figures", [])

    generated = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_verification.json"
    )
    comparisons: dict[str, bool] = {}
    reference = _reference_root()
    if reference is not None:
        pairs = [(output / "results/frozen/frozen_summary.json", reference / "frozen_summary.json")]
        pairs.extend(
            (path, reference / "tables" / path.name)
            for path in sorted((output / "results/frozen").glob("*.csv"))
        )
        comparisons = {
            str(left.relative_to(output)): _sha256(left) == _sha256(right)
            for left, right in pairs
        }
        if not all(comparisons.values()):
            failed = [name for name, passed in comparisons.items() if not passed]
            raise FetchError(f"Regenerated frozen references differ: {failed}")

    figures = sorted((output / "figures").glob("*"))
    if len(figures) != 16:
        raise FetchError(f"Expected 16 PDF/PNG figure files, found {len(figures)}")
    result = {
        "schema_version": 1,
        "profile": "analysis",
        "data_root": str(data_root),
        "generated_files": len(generated),
        "frozen_reference_comparison": comparisons,
        "figures": [path.name for path in figures],
        "hardware_jobs_submitted": 0,
    }
    (output / "analysis_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _benchmark(
    output: Path,
    cache: Path,
    *,
    smoke: bool,
    threads: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    _verify_cached("benchmark", cache)
    data_root = locate_data_root(cache)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ML4QEM_REUSE_WORKSPACE"] = str(data_root)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    result_path = output / ("benchmark_smoke.json" if smoke else "benchmark_result.json")
    predictions_path = output / (
        "benchmark_smoke_predictions.npz" if smoke else "benchmark_predictions.npz"
    )
    arguments = [
        "--input",
        str(data_root / "data/derived/local_benchmark_confirmation_v2.npz"),
        "--splits",
        str(data_root / "protocol/local_split_manifest_confirmation_v2.json"),
        "--training-size",
        "32" if smoke else "512",
        "--folds",
        *("0" if smoke else ("0", "1", "2", "3", "4")),
        "--model-seed",
        "0",
        "--threads",
        str(threads),
        "--bootstrap-draws",
        str(bootstrap_draws),
        "--output",
        str(result_path),
        "--predictions",
        str(predictions_path),
    ]
    _run_workflow("ml4qem_reuse.workflows.train_local_within_domain", arguments)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("hardware_jobs_submitted") != 0:
        raise FetchError("Benchmark result did not preserve the zero-hardware-job invariant")
    return {
        "profile": "benchmark",
        "smoke": smoke,
        "output": str(result_path),
        "predictions": str(predictions_path),
        "hardware_jobs_submitted": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml4qem-reuse")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="download and verify immutable inputs")
    fetch.add_argument("--profile", choices=("analysis", "benchmark", "full"), required=True)
    fetch.add_argument("--cache-dir", type=Path, default=None)

    reproduce = subparsers.add_parser("reproduce", help="run a reproduction profile")
    reproduce.add_argument("--profile", choices=("analysis", "benchmark", "full"), required=True)
    reproduce.add_argument("--cache-dir", type=Path, default=None)
    reproduce.add_argument("--output-dir", type=Path, default=None)
    reproduce.add_argument("--smoke", action="store_true", help="controlled one-fold benchmark test")
    reproduce.add_argument("--threads", type=int, default=1)
    reproduce.add_argument("--bootstrap-draws", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = (args.cache_dir or default_cache_dir()).resolve()
    try:
        if args.command == "fetch":
            result: Any = fetch_profile(args.profile, cache)
        else:
            output = (
                args.output_dir or Path("outputs") / args.profile
            ).expanduser().resolve()
            if args.profile == "analysis":
                if args.smoke:
                    raise FetchError("--smoke is only valid for the benchmark profile")
                result = _analysis(output, cache)
            elif args.profile == "benchmark":
                draws = args.bootstrap_draws or (128 if args.smoke else 10_000)
                if args.threads < 1 or draws < 1:
                    raise FetchError("--threads and --bootstrap-draws must be positive")
                result = _benchmark(
                    output,
                    cache,
                    smoke=args.smoke,
                    threads=args.threads,
                    bootstrap_draws=draws,
                )
            else:
                if args.smoke:
                    raise FetchError("--smoke is only valid for the benchmark profile")
                _verify_cached("full", cache)
                from ml4qem_reuse.legacy import reproduce_full

                result = reproduce_full(output, cache)
    except FetchError as exc:
        print(f"ml4qem-reuse: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
