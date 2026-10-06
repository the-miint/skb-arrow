"""What this install is (`--version`), and whether it is sound (`--doctor`)."""

import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
from typing import Any

# The libraries that compute answers (docs/capabilities.md#versioning).
COMPUTE = (
    "llvmlite",
    "numba",
    "numpy",
    "pandas",
    "patsy",
    "pyarrow",
    "scikit-bio",
    "scipy",
)
SUPPORTED = {("linux", "x86_64"), ("darwin", "arm64")}
THREADS = (
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
INSTALL = "uv tool install --managed-python --python 3.14 skb-arrow"  # the README's
_PROBLEM = "problem: "


def versions() -> Iterator[str]:
    """The package version, the protocol version, and the capabilities: a line each."""
    yield f"skb-arrow {version('skb-arrow')}"  # first: it needs no registry
    from skb_arrow import registry
    from skb_arrow.protocol import PROTOCOL_VERSION

    yield f"protocol {PROTOCOL_VERSION}"
    caps = ", ".join(f"{n}/{v}" for n, v in registry.schema_versions().items())
    yield f"capabilities: {caps}"


def _free(place: Path) -> int:
    return shutil.disk_usage(place).free


def report(
    *,
    version_info: tuple[int, ...] = tuple(sys.version_info[:3]),
    system: str = sys.platform,
    machine: str = platform.machine(),
    environ: Mapping[str, str] = os.environ,
    pins: list[str] | None = None,
    shm: Path = Path("/dev/shm"),
    free: Callable[[Path], int] = _free,
) -> int:
    """Print the facts, then a `problem:` line per fault: 1 if there is one, else 0.

    Defaults are this process's; `pins` are the installed skb-arrow's requirements.
    """
    checks: list[tuple[str, Callable[[], Iterable[str]]]] = [
        ("version", versions),
        ("python", partial(_python, version_info)),
        ("platform", partial(_platform, system, machine)),
        ("dependencies", partial(_dependencies, pins)),
        ("numpy", _numpy),
        ("numba", _numba),
        ("threads", partial(_threads, environ)),
        ("segments", partial(_segments, system, environ, shm, free)),
    ]
    facts: list[str] = []
    problems: list[str] = []
    for name, check in checks:
        try:
            for line in check():
                (problems if line.startswith(_PROBLEM) else facts).append(line)
        except Exception as e:  # a broken install still gets its report
            said = " ".join(str(e).split())  # one line, as every problem is
            problems.append(f"{_PROBLEM}{name}: {type(e).__name__}: {said}")
    print("\n".join(facts + problems))
    return 1 if problems else 0


def _python(version_info: tuple[int, ...]) -> Iterator[str]:
    shown = ".".join(map(str, version_info))
    yield f"python: {shown} {sys.executable} (base {sys.base_prefix})"
    if version_info[:2] != (3, 14):
        yield f"{_PROBLEM}python {shown} is unsupported: {INSTALL}"


def _platform(system: str, machine: str) -> Iterator[str]:
    yield f"platform: {system} {machine}"
    if (system, machine) not in SUPPORTED:
        yield (
            f"{_PROBLEM}platform {system} {machine} is unsupported: "
            "skb-arrow runs on linux x86_64 and darwin arm64"
        )


def _dependencies(pins: list[str] | None) -> Iterator[str]:
    from packaging.requirements import Requirement

    if pins is None:
        pins = requires("skb-arrow") or []
    shown: list[str] = []
    rest = 0
    problems: list[str] = []
    for requirement in map(Requirement, pins):
        name, required_by = requirement.name, f"but skb-arrow requires {requirement}"
        try:
            installed = version(name)
        except PackageNotFoundError:
            problems.append(f"{name} is not installed, {required_by}")
            continue
        if not requirement.specifier.contains(installed):
            problems.append(f"{name} {installed} is installed, {required_by}")
        if name in COMPUTE:
            shown.append(f"{name}=={installed}")
        else:
            rest += 1
    yield f"dependencies: {', '.join(sorted(shown))}; {rest} more"
    for problem in problems:
        yield f"{_PROBLEM}dependencies: {problem}"


def _numpy() -> Iterator[str]:
    import numpy

    blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
    yield f"numpy: {numpy.__version__}, BLAS {blas['name']} {blas['version']}"


def _numba() -> Iterator[str]:
    import numba

    # As configured: asking which threading layer runs would compile a kernel.
    config: Any = numba.config  # its settings are made at import, unseen by mypy
    yield (
        f"numba: {numba.__version__}, {config.NUMBA_NUM_THREADS} threads, "
        f"layer {config.THREADING_LAYER}"
    )


def _threads(environ: Mapping[str, str]) -> Iterator[str]:
    shown = (f"{v}={environ[v]}" if v in environ else f"{v} unset" for v in THREADS)
    yield f"threads: {', '.join(shown)}"


def _segments(
    system: str, environ: Mapping[str, str], shm: Path, free: Callable[[Path], int]
) -> Iterator[str]:
    """Where a caller puts DIR (docs/transport.md): it must hold one default segment."""
    from skb_arrow.transport import DEFAULT_SEGMENT_BYTES  # pyarrow, not the registry

    if system == "linux":
        place = shm
    elif system != "darwin":
        yield f"segments: no placement on {system}"  # the platform's problem says why
        return
    elif environ.get("TMPDIR"):
        place = Path(environ["TMPDIR"])
    else:
        yield "segments: TMPDIR unset"
        yield (
            f"{_PROBLEM}segments: TMPDIR is unset; on macOS a caller puts DIR under it"
        )
        return
    try:  # as a caller does
        os.rmdir(tempfile.mkdtemp(dir=place))
    except OSError as e:
        yield f"segments: {place}"
        yield f"{_PROBLEM}segments: no DIR can be created in {place}: {e.strerror}"
        return
    room = free(place)
    yield f"segments: {place}, {room >> 20} MiB free"
    if room < DEFAULT_SEGMENT_BYTES:
        advice = (
            "; place DIR elsewhere, raise Docker's --shm-size, or mount a Kubernetes "
            "emptyDir with medium: Memory"
            if system == "linux"
            else ""
        )
        yield (
            f"{_PROBLEM}segments: {place} has {room >> 20} MiB free, less than one "
            f"{DEFAULT_SEGMENT_BYTES >> 20} MiB segment, and a call's inputs and "
            f"outputs share DIR{advice}"
        )
