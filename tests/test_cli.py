import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from skb_arrow import cli, registry
from skb_arrow.capabilities import echo

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
with PYPROJECT.open("rb") as f:
    VERSION = tomllib.load(f)["project"]["version"]
# Protocol is pinned literally: bumping it must be a deliberate test edit.
EXPECTED = f"skb-arrow {VERSION}\nprotocol 1\ncapabilities: echo/1\n"


def test_version_reports_package_protocol_and_capabilities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == EXPECTED


def test_version_lists_capabilities_sorted_regardless_of_registration_order(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = {
        name: registry.Capability(schema_version, frozenset(), frozenset(), echo.run)
        for name, schema_version in [("b", 2), ("a", 1)]
    }
    monkeypatch.setattr(registry, "CAPABILITIES", registered)
    cli.main(["--version"])
    assert capsys.readouterr().out.endswith("\ncapabilities: a/1, b/2\n")


def test_installed_console_script_runs_main() -> None:
    script = Path(sys.executable).parent / "skb-arrow"
    result = subprocess.run(
        [script, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == EXPECTED


def test_bare_invocation_is_a_usage_error_without_ansi(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli.main([]) == 2
    assert capsys.readouterr().err.startswith("usage: skb-arrow")
