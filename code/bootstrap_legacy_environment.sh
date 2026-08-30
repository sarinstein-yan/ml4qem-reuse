#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${LEGACY_PYTHON:-}" ]]; then
  legacy_python="$LEGACY_PYTHON"
elif command -v python3.9 >/dev/null 2>&1; then
  legacy_python="python3.9"
elif command -v uv >/dev/null 2>&1; then
  if ! legacy_python="$(uv python find 3.9 2>/dev/null)"; then
    uv python install 3.9
    legacy_python="$(uv python find 3.9)"
  fi
else
  echo "CPython 3.9 is required; set LEGACY_PYTHON or install it with uv." >&2
  exit 1
fi
environment_dir="$project_dir/.venv-legacy"
environment_dir="${LEGACY_ENVIRONMENT_DIR:-$environment_dir}"

if [[ ! -d "$project_dir/upstream/snapshots/qiskit-community-ml-qem-9776e1b" ]]; then
  echo "The legacy bootstrap is driven by the full reproduction profile." >&2
  echo "Run 'uv run ml4qem-reuse fetch --profile full' followed by" >&2
  echo "'uv run ml4qem-reuse reproduce --profile full'." >&2
  exit 1
fi
if [[ -e "$environment_dir" ]]; then
  echo "Refusing to overwrite existing $environment_dir" >&2
  exit 1
fi

cd "$project_dir"
"$legacy_python" -m venv "$environment_dir"
"$environment_dir/bin/python" -m pip install --upgrade 'pip<26' 'setuptools<81' 'wheel==0.45.1'
"$environment_dir/bin/python" -m pip install 'numpy==1.24.4'
"$environment_dir/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.1.2+cpu'
"$environment_dir/bin/python" -m pip install \
  --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html \
  'torch-scatter==2.1.2' 'torch-sparse==0.6.18'
"$environment_dir/bin/python" -m pip install \
  -r "$project_dir/upstream/snapshots/qiskit-community-ml-qem-9776e1b/requirements.txt"
"$environment_dir/bin/python" -m pip install 'scikit-learn==1.3.2'
"$environment_dir/bin/python" -m pip install -r "$project_dir/environments/legacy-pip-freeze.txt"
"$environment_dir/bin/python" -m pip check

echo "Legacy environment created at $environment_dir"
