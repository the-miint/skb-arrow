"""What this install is (`--version`), and whether it is sound (`--doctor`)."""

import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from importlib.metadata import version
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
_INSTALL = "uv tool install --python 3.14 skb-arrow"
_PROBLEM = "problem: "


def versions() -> list[str]:
    """The package version, the protocol version, and the capabilities: a line each."""
    from skb_arrow import registry
    from skb_arrow.protocol import PROTOCOL_VERSION

    caps = ", ".join(f"{n}/{v}" for n, v in registry.schema_versions().items())
    return [
        f"skb-arrow {version('skb-arrow')}",
        f"protocol {PROTOCOL_VERSION}",
        f"capabilities: {caps}",
    ]


def report(
    *,
    version_info: tuple[int, ...] = tuple(sys.version_info[:3]),
    system: str = sys.platform,
    machine: str = platform.machine(),
    environ: Mapping[str, str] = os.environ,
    requires: list[str] | None = None,
    shm: Path = Path("/dev/shm"),
    need: int | None = None,
) -> int:
    """Print the facts, then a `problem:` line per fault: 1 if there is one, else 0.

    Defaults are this process's; `requires` is the installed skb-arrow's, `need` one
    default segment.
    """
    checks: list[tuple[str, Callable[[], Iterable[str]]]] = [
        ("version", versions),
        ("python", partial(_python, version_info)),
        ("platform", partial(_platform, system, machine)),
        ("dependencies", partial(_dependencies, requires)),
        ("numpy", _numpy),
        ("numba", _numba),
        ("threads", partial(_threads, environ)),
        ("segments", partial(_segments, system, environ, shm, need)),
    ]
    facts: list[str] = []
    problems: list[str] = []
    for name, check in checks:
        try:
            for line in check():
                (problems if line.startswith(_PROBLEM) else facts).append(line)
        except Exception as e:  # a broken install still gets its report
            problems.append(f"{_PROBLEM}{name}: {type(e).__name__}: {e}")
    print("\n".join(facts + problems))
    return 1 if problems else 0


def _python(version_info: tuple[int, ...]) -> Iterator[str]:
    shown = ".".join(map(str, version_info))
    yield f"python: {shown} {sys.executable} (base {sys.base_prefix})"
    if version_info[:2] != (3, 14):
        yield f"{_PROBLEM}python {shown} is unsupported: {_INSTALL}"


def _platform(system: str, machine: str) -> Iterator[str]:
    yield f"platform: {system} {machine}"
    if (system, machine) not in SUPPORTED:
        yield (
            f"{_PROBLEM}platform {system} {machine} is unsupported: "
            "skb-arrow runs on linux x86_64 and darwin arm64"
        )


def _dependencies(requires: list[str] | None) -> Iterator[str]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import requires as required

    from packaging.requirements import Requirement

    if requires is None:
        requires = required("skb-arrow") or []
    shown: list[str] = []
    rest = 0
    problems: list[str] = []
    for requirement in map(Requirement, requires):
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
    yield f"dependencies: {', '.join(shown)}; {rest} more"
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
    system: str, environ: Mapping[str, str], shm: Path, need: int | None
) -> Iterator[str]:
    """Where a caller puts DIR (docs/transport.md): it must hold one default segment."""
    if need is None:
        from skb_arrow.protocol import DEFAULT_SEGMENT_BYTES

        need = DEFAULT_SEGMENT_BYTES
    if system == "linux":
        place = shm
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
    free = shutil.disk_usage(place).free
    yield f"segments: {place}, {free >> 20} MiB free"
    if free < need:
        advice = (
            "; in Docker raise --shm-size, in Kubernetes use an emptyDir with "
            "medium: Memory"
            if system == "linux"
            else ""
        )
        yield (
            f"{_PROBLEM}segments: {place} has {free >> 20} MiB free, less than one "
            f"{need >> 20} MiB segment, and a call's inputs and outputs share DIR"
            f"{advice}"
        )
