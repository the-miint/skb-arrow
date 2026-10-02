import os
import select
import signal
import stat
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

# Standard library only: serve() imports the rest once the channel is reserved.

_BACKLOG = 1 << 20  # stderr held while nobody reads it; the oldest goes first
_DROPPED = "\nskb-arrow: {} bytes of stderr dropped\n"
_PARTIAL = 0.05  # seconds a line without its newline waits for it
_STALLED = 1  # seconds stderr may take no write, once the host is gone


def reserve() -> tuple[BinaryIO, int]:
    """The protocol channel (requests, response fd), moved off fds 0 and 1.

    Call first. Exec'd children don't inherit the copies. fd 0 then reads /dev/null;
    fds 1 and 2, and sys.stdout, write to a drainer that relays them to stderr without
    ever blocking the host, or to /dev/null if stderr is the channel's
    (docs/protocol.md#channel).
    """
    for fd in range(3):
        try:
            os.fstat(fd)
        except OSError:
            # Missing: fill it, or the dups below would land on it.
            os.open(os.devnull, os.O_RDWR)
    # `2>&1`: stderr is discarded, not mixed into responses.
    relay = os.open(os.devnull, os.O_WRONLY) if _same_pipe(1, 2) else _start_drainer()
    requests = os.fdopen(os.dup(0), "rb")
    responses = os.dup(1)
    host, channel = os.getpid(), (requests.fileno(), responses)

    def release() -> None:
        if os.getppid() == host:  # not a later descendant: it may reuse the numbers
            _blank(channel)

    # A fork child finds /dev/null in their place, so it can't hold the caller's EOF.
    os.register_at_fork(after_in_child=release)
    _blank((0,))
    os.dup2(relay, 1)
    os.dup2(relay, 2)
    os.close(relay)
    sys.stdout = sys.stderr
    return requests, responses


def _start_drainer() -> int:
    """Fork the drainer (DESIGN §3.13): the write end of the relay it copies to stderr.

    Before the channel is copied, so the drainer never holds it, and before any thread.
    """
    source, relay = os.pipe()
    # Forked twice: the middle process exits at once, so the drainer is never the host's
    # child, which host code reaping every child would wait on.
    if (middle := os.fork()) == 0:
        try:
            if os.fork() == 0:
                _blank((0, 1))
                # Every other fd: the relay's write end, else the relay never reaches
                # EOF, and any the caller leaked to the host, such as a channel copy.
                for fd in map(int, os.listdir("/dev/fd")):
                    if fd > 2 and fd != source:
                        with suppress(OSError):  # listdir's own, closed already
                            os.close(fd)
                _drain(source)
        except BaseException as e:
            with suppress(OSError):
                _send(2, f"skb-arrow: stderr drainer failed: {e!r}\n".encode())
            os._exit(1)
        os._exit(0)  # never back into the host's code
    os.waitpid(middle, 0)
    os.close(source)
    return relay


def _blank(fds: tuple[int, ...]) -> None:
    """Point `fds` at /dev/null."""
    null = os.open(os.devnull, os.O_RDWR)
    for fd in fds:
        os.dup2(null, fd)
    os.close(null)


def _piped(mode: int) -> bool:
    """Whether `mode` is a pipe's or a socket's."""
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def _same_pipe(a: int, b: int) -> bool:
    """Whether fds `a` and `b` are one pipe or socket."""
    x, y = os.fstat(a), os.fstat(b)
    # Inode 0 identifies nothing: macOS gives it to every TCP socket.
    same = x.st_ino != 0 and (x.st_dev, x.st_ino) == (y.st_dev, y.st_ino)
    return _piped(x.st_mode) and same


def _drain(source: int) -> None:
    """Copy `source` to fd 2 until EOF, holding at most _BACKLOG bytes unwritten."""
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal.SIG_IGN)  # to relay the host's last words
    pending = bytearray()
    dropped = 0
    done = False
    arrived = wrote = time.monotonic()  # when bytes last came in, and last went out
    ready = threading.Condition()

    def take() -> bytes | None:
        """The next piece to write, or None at the end; holding `ready`, may wait."""
        nonlocal dropped
        while True:
            if dropped:
                piece, dropped = _DROPPED.format(dropped).encode(), 0
                return piece
            # Whole lines within PIPE_BUF: atomic on a pipe others share. A longer line
            # is cut; a partial one waits a moment for its newline.
            cut = pending.rfind(b"\n", 0, select.PIPE_BUF) + 1
            wait = arrived + _PARTIAL - time.monotonic()
            if not cut and (len(pending) >= select.PIPE_BUF or done or wait <= 0):
                cut = select.PIPE_BUF
            if pending and cut:
                piece = bytes(pending[:cut])
                del pending[:cut]
                return piece
            if done:
                return None
            ready.wait(wait if pending else None)

    def write() -> None:
        # Blocking writes, in this thread only: a stalled stderr stalls nothing else.
        nonlocal wrote
        while True:
            with ready:
                piece = take()
            if piece is None:
                return
            with suppress(OSError):  # a dead stderr loses what it would show
                _send(2, piece)
            wrote = time.monotonic()

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    while chunk := os.read(source, 1 << 16):
        with ready:
            pending += chunk
            if (excess := len(pending) - _BACKLOG) > 0:
                del pending[:excess]
                dropped += excess
            arrived = time.monotonic()
            ready.notify()
    with ready:
        done = True
        ready.notify()
    # What's held goes out at stderr's pace, unless it stalls: then the drainer exits.
    wrote = time.monotonic()
    while writer.is_alive() and (left := wrote + _STALLED - time.monotonic()) > 0:
        writer.join(left)


def serve(directory: Path, requests: BinaryIO, responses: int) -> int:
    """Answer requests until EOF, then clean `directory`: the exit status."""
    if problem := _unusable(directory):
        print(f"skb-arrow: --segment-dir {directory}: {problem}", file=sys.stderr)
        return 2
    # Absolute and link-free, so a later chdir can't redirect cleanup.
    directory = directory.resolve()
    host = os.getpid()
    try:
        from skb_arrow import protocol  # loads pyarrow and every capability

        session = protocol.Session(directory)
        _watch(responses, session.lock, directory)
        for line in requests:
            _send(responses, protocol.handle(session, line))
    except BrokenPipeError:
        return 1
    finally:
        if os.getpid() == host:  # not a fork child unwinding into this frame
            _clean(directory)
    return 0


def _watch(responses: int, lock: threading.Lock, directory: Path) -> None:
    """Once `responses` has no reader and `lock` is free: clean `directory`, exit 1.

    The session holds `lock` except while a capability runs (DESIGN §3.14).
    """
    if not _piped(os.fstat(responses).st_mode):
        return  # a file has no reader to lose

    def watch() -> None:
        _await_no_reader(responses)
        lock.acquire()  # never released: the session writes nothing more
        try:
            _clean(directory)
        except OSError as e:  # the caller may be gone; its stderr may not
            print(f"skb-arrow: cleaning {directory}: {e}", file=sys.stderr, flush=True)
        finally:
            os._exit(1)

    threading.Thread(target=watch, name="skb-arrow watch", daemon=True).start()


def _await_no_reader(fd: int) -> None:
    """Return once pipe or socket `fd` has no reader."""
    if sys.platform == "darwin":  # its poll reports nothing unasked
        queue = select.kqueue()
        flags = select.KQ_EV_ADD | select.KQ_EV_CLEAR  # on a change, not while writable
        try:
            queue.control([select.kevent(fd, select.KQ_FILTER_WRITE, flags)], 0)
        except BrokenPipeError:  # gone before the watch began
            return
        while not any(e.flags & select.KQ_EV_EOF for e in queue.control(None, 1)):
            pass
    else:
        watch = select.poll()
        watch.register(fd, 0)  # POLLERR and POLLHUP come unasked
        watch.poll()


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
