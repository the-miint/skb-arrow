"""skb-arrow --doctor (docs/architecture.md#cli)."""

import os
import shutil
import subprocess
import sys
from importlib.metadata import requires, version
from pathlib import Path
from typing import Any

import numba
import numpy
import pytest
from client import SKB_ARROW
from test_cli import EXPECTED

from skb_arrow import doctor

KEYS = ["python", "platform", "dependencies", "numpy", "numba", "threads", "segments"]
SHARE = "less than one 256 MiB segment, and a call's inputs and outputs share DIR"
ADVICE = (
    "; in Docker raise --shm-size, in Kubernetes use an emptyDir with medium: Memory"
)


@pytest.fixture
def healthy(tmp_path: Path) -> dict[str, Any]:
    """A sound Linux x86_64 install's inputs, segments placed in tmp_path."""
    return {
        "version_info": (3, 14, 0),
        "system": "linux",
        "machine": "x86_64",
        "environ": {"TMPDIR": str(tmp_path)},
        "shm": tmp_path,
    }


def run(capsys: pytest.CaptureFixture[str], **inputs: Any) -> tuple[int, list[str]]:
    status = doctor.report(**inputs)
    return status, capsys.readouterr().out.splitlines()


def problems(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("problem: ")]


def keys(lines: list[str]) -> list[str]:
    return [line.split(":")[0] for line in lines if not line.startswith("problem: ")]


def test_a_sound_install_reports_its_facts_and_exits_0(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any], tmp_path: Path
) -> None:
    status, lines = run(capsys, **healthy)
    assert (status, problems(lines)) == (0, [])
    assert lines[:3] == EXPECTED.splitlines()  # --version's
    assert keys(lines[3:]) == KEYS
    pinned = requires("skb-arrow") or []
    blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
    config: Any = numba.config  # its settings are made at import
    assert lines[3:-1] == [
        f"python: 3.14.0 {sys.executable} (base {sys.base_prefix})",
        "platform: linux x86_64",
        "dependencies: "
        + ", ".join(f"{name}=={version(name)}" for name in doctor.COMPUTE)
        + f"; {len(pinned) - len(doctor.COMPUTE)} more",
        f"numpy: {numpy.__version__}, BLAS {blas['name']} {blas['version']}",
        f"numba: {numba.__version__}, {config.NUMBA_NUM_THREADS} threads, "
        f"layer {config.THREADING_LAYER}",
        "threads: OMP_NUM_THREADS unset, NUMBA_NUM_THREADS unset, "
        "OPENBLAS_NUM_THREADS unset, VECLIB_MAXIMUM_THREADS unset",
    ]
    assert lines[-1].startswith(f"segments: {tmp_path}, ")
    assert lines[-1].endswith(" MiB free")


@pytest.mark.parametrize("version_info", [(3, 13, 9), (3, 15, 0)])
def test_python_other_than_3_14_is_a_problem(
    capsys: pytest.CaptureFixture[str],
    healthy: dict[str, Any],
    version_info: tuple[int, ...],
) -> None:
    status, lines = run(capsys, **healthy | {"version_info": version_info})
    shown = ".".join(map(str, version_info))
    assert status == 1
    assert problems(lines) == [
        f"problem: python {shown} is unsupported: "
        "uv tool install --python 3.14 skb-arrow"
    ]
    assert lines[-1] == problems(lines)[0]  # after every fact, though found second


@pytest.mark.parametrize(
    ("system", "machine"),
    [("linux", "aarch64"), ("darwin", "x86_64"), ("win32", "AMD64")],
)
def test_an_unsupported_platform_is_a_problem(
    capsys: pytest.CaptureFixture[str],
    healthy: dict[str, Any],
    system: str,
    machine: str,
) -> None:
    status, lines = run(capsys, **healthy | {"system": system, "machine": machine})
    assert status == 1
    assert problems(lines) == [
        f"problem: platform {system} {machine} is unsupported: "
        "skb-arrow runs on linux x86_64 and darwin arm64"
    ]


def test_macos_arm64_is_supported_with_segments_under_tmpdir(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any], tmp_path: Path
) -> None:
    darwin = {"system": "darwin", "machine": "arm64", "shm": tmp_path / "missing"}
    status, lines = run(capsys, **healthy | darwin)
    assert (status, problems(lines)) == (0, [])
    assert "platform: darwin arm64" in lines
    assert lines[-1].startswith(f"segments: {tmp_path}, ")


def test_linux_places_segments_in_shm_whatever_tmpdir_says(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any], tmp_path: Path
) -> None:
    for environ in [{}, {"TMPDIR": str(tmp_path / "missing")}]:
        status, lines = run(capsys, **healthy | {"environ": environ})
        assert (status, problems(lines)) == (0, [])
        assert lines[-1].startswith(f"segments: {tmp_path}, ")


@pytest.mark.parametrize("tmpdir", [None, ""])
def test_tmpdir_unset_on_macos_is_a_problem(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any], tmpdir: str | None
) -> None:
    environ = {} if tmpdir is None else {"TMPDIR": tmpdir}
    darwin = {"system": "darwin", "machine": "arm64", "environ": environ}
    status, lines = run(capsys, **healthy | darwin)
    assert status == 1
    assert "segments: TMPDIR unset" in lines
    assert problems(lines) == [
        "problem: segments: TMPDIR is unset; on macOS a caller puts DIR under it"
    ]


def test_a_dependency_its_pin_excludes_is_a_problem(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any]
) -> None:
    status, lines = run(capsys, **healthy | {"requires": ["numpy==0.0.1"]})
    assert status == 1
    assert problems(lines) == [
        f"problem: dependencies: numpy {numpy.__version__} is installed, "
        "but skb-arrow requires numpy==0.0.1"
    ]


def test_a_missing_dependency_is_a_problem(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any]
) -> None:
    status, lines = run(capsys, **healthy | {"requires": ["no-such-dist==1.0"]})
    assert status == 1
    assert problems(lines) == [
        "problem: dependencies: no-such-dist is not installed, "
        "but skb-arrow requires no-such-dist==1.0"
    ]


def test_dependencies_shows_compute_pins_and_counts_the_rest(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any]
) -> None:
    pins = [f"six=={version('six')}", f"numpy=={numpy.__version__}"]
    status, lines = run(capsys, **healthy | {"requires": pins})
    assert status == 0
    assert f"dependencies: numpy=={numpy.__version__}; 1 more" in lines


def test_threads_shows_each_variable_set_or_unset(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any]
) -> None:
    environ = healthy["environ"] | {
        "OMP_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    status, lines = run(capsys, **healthy | {"environ": environ})
    assert status == 0
    assert (
        "threads: OMP_NUM_THREADS=2, NUMBA_NUM_THREADS unset, "
        "OPENBLAS_NUM_THREADS unset, VECLIB_MAXIMUM_THREADS=1"
    ) in lines


@pytest.mark.parametrize(("system", "advice"), [("linux", ADVICE), ("darwin", "")])
def test_less_room_than_a_segment_is_a_problem(
    capsys: pytest.CaptureFixture[str],
    healthy: dict[str, Any],
    tmp_path: Path,
    system: str,
    advice: str,
) -> None:
    free = shutil.disk_usage(tmp_path).free
    inputs = healthy | {
        "system": system,
        "machine": "x86_64" if system == "linux" else "arm64",
    }
    assert run(capsys, **inputs | {"need": free // 2})[0] == 0
    status, lines = run(capsys, **inputs | {"need": 2 * free})
    assert status == 1
    [problem] = problems(lines)
    assert problem.startswith(f"problem: segments: {tmp_path} has ")
    assert problem.endswith(
        f" MiB free, less than one {2 * free >> 20} MiB segment, and a call's inputs "
        f"and outputs share DIR{advice}"
    )


def test_a_place_no_dir_can_be_created_in_is_a_problem(
    capsys: pytest.CaptureFixture[str], healthy: dict[str, Any], tmp_path: Path
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        status, lines = run(capsys, **healthy | {"shm": locked})
    finally:
        locked.chmod(0o700)
    assert status == 1
    assert f"segments: {locked}" in lines
    assert problems(lines) == [
        f"problem: segments: no DIR can be created in {locked}: Permission denied"
    ]
    status, lines = run(capsys, **healthy | {"shm": tmp_path / "missing"})
    assert status == 1
    assert problems(lines) == [
        f"problem: segments: no DIR can be created in {tmp_path / 'missing'}: "
        "No such file or directory"
    ]


def test_a_check_that_raises_is_a_problem_not_a_traceback() -> None:
    code = (
        "import sys\n"
        "sys.modules['skb_arrow.registry'] = None  # as a broken numba would\n"
        "from skb_arrow import cli\n"
        "sys.exit(cli.main(['--doctor']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert (
        "problem: version: ModuleNotFoundError: import of skb_arrow.registry halted; "
        "None in sys.modules"
    ) in lines
    assert keys(lines)[:6] == KEYS[:6]  # the other checks still report


def test_the_installed_console_script_reports_numba_as_configured() -> None:
    environ = os.environ | {
        "NUMBA_NUM_THREADS": "2",
        "NUMBA_THREADING_LAYER": "workqueue",
    }
    result = subprocess.run(
        [*SKB_ARROW, "--doctor"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=environ,
    )
    lines = result.stdout.splitlines()
    # Not exit 0: this machine's placement may be small (a container's /dev/shm).
    assert result.returncode == (1 if problems(lines) else 0)
    assert result.stderr == ""
    assert lines[:3] == EXPECTED.splitlines()
    assert keys(lines[3:]) == KEYS
    assert lines[len(lines) - len(problems(lines)) :] == problems(lines)  # last
    assert f"numba: {numba.__version__}, 2 threads, layer workqueue" in lines
