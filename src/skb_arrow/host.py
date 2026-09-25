import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

# Standard library only: serve() imports the rest once the channel is reserved.


def reserve() -> tuple[BinaryIO, int]:
    """The protocol channel (requests, response fd), moved off fds 0 and 1.

    Call first. Children don't inherit the copies. fd 0 then reads /dev/null; fd 1
    and sys.stdout write to stderr, line-buffered, so diagnostics survive a kill.
    """
    requests = os.fdopen(os.dup(0), "rb")
    responses = os.dup(1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return requests, responses


def serve(directory: Path, requests: BinaryIO, responses: int) -> int:
    """Answer requests until EOF, then clean `directory`: the exit status."""
    if problem := _unusable(directory):
        print(f"skb-arrow: --segment-dir {directory}: {problem}", file=sys.stderr)
        return 2
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
    if any(directory.iterdir()):
        return "is not empty"
    if not os.access(directory, os.W_OK | os.X_OK):
        return "is not writable"
    return None


def _send(fd: int, data: bytes) -> None:
    # Unbuffered: nothing is left to flush into a closed pipe at exit.
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _clean(directory: Path) -> None:
    """Unlink `directory`'s entries, then remove it; what's gone is done."""
    with suppress(FileNotFoundError):
        with os.scandir(directory) as entries:
            for entry in entries:
                with suppress(FileNotFoundError):
                    os.unlink(entry.path)
        os.rmdir(directory)
