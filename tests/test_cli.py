import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from client import SKB_ARROW

from skb_arrow import cli, registry
from skb_arrow.capabilities import echo

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
with PYPROJECT.open("rb") as f:
    VERSION = tomllib.load(f)["project"]["version"]
# Protocol is pinned literally: bumping it must be a deliberate test edit.
CAPABILITIES = ", ".join(f"{n}/{v}" for n, v in registry.schema_versions().items())
EXPECTED = f"skb-arrow {VERSION}\nprotocol 1\ncapabilities: {CAPABILITIES}\n"


def test_version_reports_package_protocol_and_capabilities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == EXPECTED


def test_version_lists_capabilities_sorted_regardless_of_registration_order(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = {
        name: registry.Capability(schema_version, frozenset(), {}, echo.run)
        for name, schema_version in [("b", 2), ("a", 1)]
    }
    monkeypatch.setattr(registry, "CAPABILITIES", registered)
    cli.main(["--version"])
    assert capsys.readouterr().out.endswith("\ncapabilities: a/1, b/2\n")


def test_installed_console_script_runs_main() -> None:
    result = subprocess.run(
        [*SKB_ARROW, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == EXPECTED


def test_importing_the_cli_and_host_loads_nothing_heavy() -> None:
    # Anything heavy imported before host.reserve() could print to the channel.
    code = (
        "import sys, skb_arrow.cli, skb_arrow.host\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in"
        " ('pyarrow', 'skb_arrow')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.stdout == "['skb_arrow', 'skb_arrow.cli', 'skb_arrow.host']\n"


def test_version_and_segment_dir_are_exclusive() -> None:
    # In a subprocess: serving in-process would reserve pytest's own fds.
    result = subprocess.run(
        [*SKB_ARROW, "--version", "--segment-dir", "x"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_bare_invocation_is_a_usage_error_without_ansi(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli.main([]) == 2
    assert capsys.readouterr().err.startswith("usage: skb-arrow")
