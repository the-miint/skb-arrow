import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from skb_arrow import cli, registry

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
VERSION = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
# Protocol is pinned literally: bumping it must be a deliberate test edit.
EXPECTED = f"skb-arrow {VERSION}\nprotocol 1\ncapabilities: none\n"


def test_version_reports_package_protocol_and_empty_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == EXPECTED


def test_version_lists_capabilities_sorted_regardless_of_registration_order(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(registry.CAPABILITIES, "b", 2)
    monkeypatch.setitem(registry.CAPABILITIES, "a", 1)
    cli.main(["--version"])
    assert capsys.readouterr().out.endswith("\ncapabilities: a/1, b/2\n")


def test_installed_console_script_runs_main() -> None:
    script = Path(sys.executable).parent / "skb-arrow"
    result = subprocess.run([script, "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == EXPECTED


def test_bare_invocation_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert capsys.readouterr().err.startswith("usage: skb-arrow")
