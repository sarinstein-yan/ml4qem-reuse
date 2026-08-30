"""Standards-compliant serialization helpers for scientific result records."""

from __future__ import annotations

import json
import math
from typing import Any


def finite_interval_fields(width: float | None) -> dict[str, bool | float | None]:
    """Encode a finite interval width or an explicit finite-interval abstention."""

    numeric_width = None if width is None else float(width)
    available = numeric_width is not None and math.isfinite(numeric_width)
    return {
        "finite_interval_available": available,
        "interval_mean_width": numeric_width if available else None,
    }


def finite_interval_metric_fields(
    width: float | None, coverage_fields: dict[str, float | None]
) -> dict[str, bool | float | None]:
    """Encode width and coverage only when a finite interval is available."""

    encoded = finite_interval_fields(width)
    available = bool(encoded["finite_interval_available"])
    encoded.update(
        {
            key: float(value) if available and value is not None else None
            for key, value in coverage_fields.items()
        }
    )
    return encoded


def migrate_finite_interval_fields(value: Any) -> int:
    """Replace non-finite interval widths in a nested record in place.

    Returns the number of interval records visited. The final strict JSON dump
    remains responsible for rejecting any unrelated non-finite value.
    """

    visited = 0
    if isinstance(value, dict):
        if "interval_mean_width" in value:
            width = value["interval_mean_width"]
            coverage_fields = {
                key: child
                for key, child in value.items()
                if key
                in {
                    "interval_marginal_coverage",
                    "interval_row_joint_coverage",
                    "interval_condition_circuit_joint_coverage",
                }
            }
            encoded = finite_interval_metric_fields(
                None if width is None else float(width), coverage_fields
            )
            value.update(encoded)
            visited += 1
        for child in value.values():
            visited += migrate_finite_interval_fields(child)
    elif isinstance(value, list):
        for child in value:
            visited += migrate_finite_interval_fields(child)
    return visited


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize JSON while rejecting RFC 8259-incompatible numeric tokens."""

    return json.dumps(value, allow_nan=False, **kwargs)
