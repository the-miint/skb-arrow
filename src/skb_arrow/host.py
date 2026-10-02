import os
import select
import signal
import stat
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

# Standard library only: serve() imports the rest once the channel is reserved.

_BACKLOG = 1 << 20  # stderr held while nobody reads it; the oldest goes first
_DROPPED = "\nskb-arrow: {} bytes of stderr dropped\n"


def reserve() -> tuple[BinaryIO, int]:
    """The protocol channel (requests, response fd), moved off fds 0 and 1.

    Call first. Exec'd children don't inherit the copies. fd 0 then reads /dev/null;
    fds 1 and 2, and sys.stdout, write to a drainer that relays them to stderr without
    ever blocking the host (docs/protocol.md#channel).
    """
    for fd in range(3):
        try:
            os.fstat(fd)
        except OSError:
            # Missing: fill it, or the dups below would land on it.
            os.open(os.devnull, os.O_RDWR)
    relay = _start_drainer()
    requests = os.fdopen(os.dup(0), "rb")
    responses = os.dup(1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(relay, 1)
    os.dup2(relay, 2)
    os.close(relay)
    sys.stdout = sys.stderr
    return requests, responses


def _start_drainer() -> int:
    """Fork the drainer (DESIGN §3.13): the write end of the relay it copies to stderr.

    Before the channel is copied, so the drainer never holds it, and before any thread.
    """
    into_channel = _same_pipe(1, 2)  # `2>&1`
    source, relay = os.pipe()
    if os.fork() == 0:
        try:
            os.close(relay)  # else the relay never reaches EOF
            null = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2) if into_channel else (0, 1):
                os.dup2(null, fd)
            os.close(null)
            _drain(source)
        finally:
            os._exit(0)  # never back into the host's code
    os.close(source)
    return relay


def _same_pipe(a: int, b: int) -> bool:
    """Whether fds `a` and `b` are one pipe or socket."""
    x, y = os.fstat(a), os.fstat(b)
    piped = stat.S_ISFIFO(x.st_mode) or stat.S_ISSOCK(x.st_mode)
    return piped and (x.st_dev, x.st_ino) == (y.st_dev, y.st_ino)


def _drain(source: int) -> None:
    """Copy `source` to fd 2 until EOF, holding at most _BACKLOG bytes unwritten."""
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal.SIG_IGN)  # to relay the host's last words
    pending = bytearray()
    dropped = 0
    done = False
    ready = threading.Condition()

    def due() -> bool:
        return bool(pending or dropped or done)

    def write() -> None:
        # Blocking writes, in this thread only: a stalled stderr stalls nothing else.
        nonlocal dropped
        while True:
            with ready:
                ready.wait_for(due)
                if dropped:
                    piece, dropped = _DROPPED.format(dropped).encode(), 0
                elif pending:
                    # Whole lines within PIPE_BUF: atomic on a pipe others share.
                    cut = (
                        pending.rfind(b"\n", 0, select.PIPE_BUF) + 1 or select.PIPE_BUF
                    )
                    piece = bytes(pending[:cut])
                    del pending[:cut]
                else:
                    return
            with suppress(OSError):  # a dead stderr loses what it would show
                view = memoryview(piece)
                while view:
                    view = view[os.write(2, view) :]

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    while chunk := os.read(source, 1 << 16):
        with ready:
            pending += chunk
            if (excess := len(pending) - _BACKLOG) > 0:
                del pending[:excess]
                dropped += excess
            ready.notify()
    with ready:
        done = True
        ready.notify()
    writer.join(1)  # a stalled stderr can't keep the drainer alive


def serve(directory: Path, requests: BinaryIO, responses: int) -> int:
    """Answer requests until EOF, then clean `directory`: the exit status."""
    if problem := _unusable(directory):
        print(f"skb-arrow: --segment-dir {directory}: {problem}", file=sys.stderr)
        return 2
    # Absolute and link-free, so a later chdir can't redirect cleanup.
    directory = directory.resolve()
    try:
        from skb_arrow import protocol  # loads pyarrow and every capability

        session = protocol.Session(directory)
        for line in requests:
            _send(responses, protocol.handle(session, line))
    except BrokenPipeError:
        return 1
    finally:
        _clean(directory)
    return 0


def _unusable(directory: Path) -> str | None:
    """Why `directory` can't be a session directory, if it can't."""
    if not directory.exists():
        return "does not exist"
    if not directory.is_dir():
        return "is not a directory"
    if not os.access(directory, os.R_OK | os.X_OK):
        return "is not readable"
    if not os.access(directory, os.W_OK):
        return "is not writable"
    if any(directory.iterdir()):
        return "is not empty"
    return None


def _send(fd: int, data: bytes) -> None:
    # Unbuffered: nothing is left to flush into a closed pipe at exit.
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _clean(directory: Path) -> None:
    """Unlink `directory`'s entries, then remove it; what's gone is done.

    Every removal is tried; the first failure is raised after.
    """
    failures: list[OSError] = []

    def remove(unlink: Callable[[str], None], path: str) -> None:
        try:
            unlink(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            failures.append(e)

    with suppress(FileNotFoundError), os.scandir(directory) as entries:
        for entry in entries:
            remove(os.unlink, entry.path)
    remove(os.rmdir, str(directory))
    if failures:
        raise failures[0]
