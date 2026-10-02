"""A host with test capabilities: fixture_host.py --segment-dir DIR."""

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from skb_arrow import host

if TYPE_CHECKING:
    import pyarrow as pa


DYING = threading.Event()


def noisy(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    os.chdir("/")  # a relative DIR must not follow
    print("print during run")
    os.write(1, b"write during run\n")
    subprocess.run(["echo", "child during run"], check=True)
    sys.stdin.read()  # would swallow the pipelined requests if stdin were the channel
    return tables["table"]


def fail(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    from skb_arrow.errors import Unsupported

    failures = {
        "unsupported": Unsupported("extra missing"),
        "resource": MemoryError(),
        "internal": RuntimeError("bug"),
    }
    raise failures[str(params["kind"])]


def flood(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    """`bytes` bytes of stderr, written as `how` says, then the last words."""
    size, how = int(str(params["bytes"])), params["how"]
    if how == "print":
        print("x" * (size - 1))
    elif how in ("fd1", "fd2", "lines"):
        data = b"x" * size if how != "lines" else (b"y" * 99 + b"\n") * (size // 100)
        view = memoryview(data)
        while view:
            view = view[os.write(1 if how == "fd1" else 2, view) :]
    elif how == "gil":  # PyDLL holds the GIL through the call
        import ctypes  # here: it holds a file open, which could take a closed fd 2

        ctypes.PyDLL(None).write(2, b"x" * size, size)
    elif how == "trickle":  # one line, in 30 writes over 0.6 s
        for i in range(30):
            os.write(2, b"x" * (size * (i + 1) // 30 - size * i // 30))
            time.sleep(0.02)
    elif how == "halves":  # one line in two writes, a moment apart
        os.write(2, b"z" * (size // 2))
        time.sleep(0.01)
        os.write(2, b"z" * (size - size // 2 - 1) + b"\n")
    elif how == "child":
        subprocess.run(["head", "-c", str(size), "/dev/zero"], stdout=2, check=True)
    print("last words")
    return tables["table"]


def lingers(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    subprocess.Popen(["sleep", "5"])  # holds fds 1 and 2, as an exec'd child may
    return tables["table"]


def mutters(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    """A line without its newline; with `dies`, the host's last."""
    os.write(2, b"no newline")
    if params["dies"]:
        os.kill(os.getpid(), signal.SIGKILL)
    return tables["table"]


def childless(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    """Fails if the host has a child, live or a zombie: code reaping all would wait."""
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        return tables["table"]
    raise RuntimeError("the host has a child")


def dies(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    print("about to die")
    DYING.set()
    return tables["table"]


def broken(source: int) -> None:
    raise RuntimeError("broken on purpose")


def main() -> int:
    if "FIXTURE_BREAK_DRAINER" in os.environ:
        host._drain = broken
    channel = host.reserve()
    # As a capability's library might, on import.
    print("import-time print")
    os.write(1, b"import-time write\n")

    from skb_arrow import protocol, registry

    table = frozenset({"table"})
    registry.CAPABILITIES |= {
        "noisy": registry.Capability(1, table, {}, noisy),
        "fail": registry.Capability(1, table, {"kind": registry.Param(str)}, fail),
        "dies": registry.Capability(1, table, {}, dies),
        "flood": registry.Capability(
            1, table, {"how": registry.Param(str), "bytes": registry.Param(int)}, flood
        ),
        "lingers": registry.Capability(1, table, {}, lingers),
        "childless": registry.Capability(1, table, {}, childless),
        "mutters": registry.Capability(
            1, table, {"dies": registry.Param(bool)}, mutters
        ),
    }
    handle = protocol.handle

    def killed_before_replying(session: protocol.Session, line: bytes) -> bytes:
        response = handle(session, line)
        if DYING.is_set():
            os.kill(os.getpid(), signal.SIGKILL)
        return response

    protocol.handle = killed_before_replying
    return host.serve(Path(sys.argv[-1]), *channel)


if __name__ == "__main__":
    sys.exit(main())
