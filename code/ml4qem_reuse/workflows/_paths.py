"""Resolve the data workspace without embedding a machine-local path."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the active data workspace or the source repository root."""

    configured = os.environ.get("ML4QEM_REUSE_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def output_root() -> Path:
    """Return the requested output root, defaulting to the active workspace."""

    configured = os.environ.get("ML4QEM_REUSE_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_root()

