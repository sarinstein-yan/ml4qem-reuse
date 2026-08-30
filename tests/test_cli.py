import pytest

from ml4qem_reuse.cli import _verify_cached, build_parser
from ml4qem_reuse.fetcher import FetchError


def test_required_public_commands_parse() -> None:
    parser = build_parser()
    assert parser.parse_args(["fetch", "--profile", "analysis"]).profile == "analysis"
    for profile in ("analysis", "benchmark", "full"):
        parsed = parser.parse_args(["reproduce", "--profile", profile])
        assert parsed.profile == profile


def test_benchmark_smoke_is_explicit() -> None:
    parsed = build_parser().parse_args(
        ["reproduce", "--profile", "benchmark", "--smoke", "--threads", "1"]
    )
    assert parsed.smoke is True
    assert parsed.threads == 1


def test_reproduce_without_fetch_points_to_fetch_command(tmp_path) -> None:
    with pytest.raises(FetchError, match="fetch --profile benchmark"):
        _verify_cached("benchmark", tmp_path)
