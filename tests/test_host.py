import json
import os
import pty
import re
import select
import signal
import socket
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from client import FIXTURE, INIT, SKB_ARROW, Client

from skb_arrow import host, registry

TABLE = pa.table({"a": pa.array(range(1000), pa.int64())})


def call(capability: str, name: str, **params: object) -> dict[str, object]:
    return {
        "type": "call",
        "capability": capability,
        "params": params,
        "input": {"table": [name]},
    }


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    directory = tmp_path / "session"
    directory.mkdir()
    return directory


@pytest.fixture
def connect(directory: Path) -> Iterator[Callable[..., Client]]:
    """Start clients; any host still running at teardown is killed."""
    clients: list[Client] = []

    def start(command: list[str], where: Path = directory, **popen: Any) -> Client:
        clients.append(Client(command, where, **popen))
        return clients[-1]

    yield start
    for client in clients:
        client.kill()


def test_echo_round_trips_segments_and_shutdown_removes_dir(
    connect: Callable[..., Client], directory: Path
) -> None:
    client = connect(SKB_ARROW)
    ready = client.send(INIT | {"segment_bytes": 1024})
    assert ready["capabilities"] == registry.schema_versions()
    names = [client.put(f"in-{i}", TABLE.slice(i * 250, 250)) for i in range(4)]
    result = client.send(call("echo", "unused") | {"input": {"table": names}})
    assert len(result["output"]) == 8
    assert client.fetch(result["output"]).equals(TABLE)
    # Left for the host: an output never fetched, and a segment never named.
    assert client.send(call("echo", client.put("a", TABLE)))["type"] == "result"
    client.put("unnamed", TABLE)
    assert client.shut() == 0
    assert not directory.exists()


def test_every_kind_end_to_end(connect: Callable[..., Client], directory: Path) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    cases = [
        call("nope", client.put("a", TABLE)),
        call("echo", client.put("b", b"not arrow")),
        call("echo", client.put("c", TABLE), x=1),
        *(
            call("fail", client.put(k, TABLE), kind=k)
            for k in ["unsupported", "resource", "internal"]
        ),
    ]
    kinds = [client.send(message)["kind"] for message in cases]
    assert kinds == [
        "host_incompatible",
        "invalid_input",
        "invalid_param",
        "unsupported",
        "resource",
        "internal",
    ]
    assert client.send(call("echo", client.put("d", TABLE)))["type"] == "result"
    assert client.shut() == 0


def test_the_channel_stays_clean_whatever_prints(
    connect: Callable[..., Client], directory: Path
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    # Pipelined: the second request is already waiting while `noisy` reads stdin.
    client.write(call("noisy", client.put("a", TABLE)))
    client.write(call("echo", client.put("b", TABLE)))
    assert [client.receive()["type"] for _ in range(2)] == ["result", "result"]
    assert client.shut() == 0
    log = client.stderr()
    noises = ["import-time print", "import-time write", "print during run"]
    noises += ["write during run", "child during run"]
    assert [noise for noise in noises if noise not in log] == []


def test_a_killed_host_leaves_outputs_only_in_dir_for_the_caller_to_clean(
    tmp_path: Path, connect: Callable[..., Client], directory: Path
) -> None:
    private = tmp_path / "tmp"
    private.mkdir()
    client = connect(FIXTURE, env=os.environ | {"TMPDIR": str(private)})
    client.send(INIT)
    client.write(call("dies", client.put("a", TABLE)))
    assert client.shut() == -signal.SIGKILL
    assert [p.name for p in directory.iterdir()] == ["skbout-0"]
    assert list(private.iterdir()) == []
    assert "about to die" in client.stderr()
    client.clean()
    assert not directory.exists()


def test_a_caller_that_stops_reading_is_cleaned_up_after(
    connect: Callable[..., Client], directory: Path
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    assert client.process.stdout
    client.process.stdout.close()
    client.write(call("echo", client.put("a", TABLE)))
    assert client.process.wait(timeout=30) == 1
    assert not directory.exists()
    assert "Traceback" not in client.stderr()


@pytest.mark.parametrize("channel", ["pipe", "socket"])
def test_a_caller_that_stops_reading_mid_call_stops_the_host(
    connect: Callable[..., Client], directory: Path, channel: str
) -> None:
    ours, theirs = socket.socketpair()
    client = connect(
        FIXTURE, stdout=theirs.fileno() if channel == "socket" else subprocess.PIPE
    )
    theirs.close()
    responses = client.process.stdout or ours.makefile("rb")
    client.write(INIT)
    assert json.loads(responses.readline())["type"] == "ready"
    client.write(call("stall", client.put("a", TABLE)))
    client.put("b", TABLE)  # for a later call: DIR holds more than the call's own
    responses.close()
    ours.close()
    with suppress(subprocess.TimeoutExpired):
        client.process.wait(timeout=5)
    assert client.process.returncode == 1  # within 5 s, not after the 60 s stall
    assert not directory.exists()


@pytest.mark.parametrize("first", ["stdin", "stdout"])
def test_shutdown_exits_0_whichever_pipe_closes_first(
    connect: Callable[..., Client], directory: Path, first: str
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    assert client.send(call("echo", client.put("a", TABLE)))["type"] == "result"
    assert client.process.stdin and client.process.stdout
    pipes = [client.process.stdin, client.process.stdout]
    for pipe in pipes if first == "stdin" else pipes[::-1]:
        pipe.close()
        time.sleep(0.2)  # for a watch that acts while idle to act
    assert client.process.wait(timeout=10) == 0
    assert not directory.exists()


def test_responses_to_a_file_are_not_watched(tmp_path: Path, directory: Path) -> None:
    requests = [INIT, {"type": "call", "capability": "threads", "input": {}}]
    with open(tmp_path / "responses", "wb") as responses:
        result = subprocess.run(
            [*FIXTURE, "--segment-dir", str(directory)],
            input=b"".join(json.dumps(r).encode() + b"\n" for r in requests),
            stdout=responses,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    assert result.returncode == 0
    assert b"threads: ['MainThread']" in result.stderr


def test_an_idle_host_does_not_spin(connect: Callable[..., Client]) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    names = iter(range(2))

    def cpu() -> float:
        result = client.send(call("cpu", client.put(f"c{next(names)}", TABLE)))
        return float(client.fetch(result["output"])["seconds"][0].as_py())

    before = cpu()
    time.sleep(2)
    assert cpu() - before < 0.5


def test_cleanup_tries_every_entry_before_failing(
    connect: Callable[..., Client], directory: Path
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    for i in range(20):
        client.put(f"s{i}", b"")
    (directory / "sub").mkdir()  # breaks the rules; unlink can't remove it
    assert client.shut() == 1
    assert [p.name for p in directory.iterdir()] == ["sub"]


def test_a_symlinked_dir_is_cleaned_at_its_target(
    tmp_path: Path, connect: Callable[..., Client]
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    client = connect(FIXTURE, link)
    client.send(INIT)
    client.put("a", TABLE)
    assert client.shut() == 0
    assert not target.exists()


def test_the_channel_survives_a_closed_stderr(connect: Callable[..., Client]) -> None:
    client = connect(["sh", "-c", 'exec "$0" "$@" 2>&-', *FIXTURE])
    client.send(INIT)
    assert client.send(call("noisy", client.put("a", TABLE)))["type"] == "result"
    assert client.shut() == 0


FLOOD = 4 << 20  # far past any pipe's buffer
DROPPED = re.compile(rb"\nskb-arrow: (\d+) bytes of stderr dropped\n")


def emptied(group: int, within: float) -> bool:
    """Whether process group `group` has no members left within `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def read_to_eof(fd: int, within: float, size: int = 1 << 16, pause: float = 0) -> bytes:
    """Pipe `fd` to EOF, `size` bytes a read, `pause` s apart; fails past `within` s."""
    data, deadline = bytearray(), time.monotonic() + within
    while (left := deadline - time.monotonic()) > 0:
        if select.select([fd], [], [], left)[0]:
            if not (chunk := os.read(fd, size)):
                return bytes(data)
            data += chunk
            time.sleep(pause)
    pytest.fail(f"no EOF within {within} s")


@pytest.mark.parametrize("how", ["print", "fd1", "fd2", "gil", "child"])
@pytest.mark.parametrize("sink", ["pipe", "pty"])
def test_an_unread_stderr_cannot_wedge_the_host(
    connect: Callable[..., Client], how: str, sink: str
) -> None:
    unread, stderr = os.pipe() if sink == "pipe" else pty.openpty()
    try:
        client = connect(FIXTURE, timeout=10, stderr=stderr)
        os.close(stderr)
        client.send(INIT)
        flood = call("flood", client.put("a", TABLE), how=how, bytes=FLOOD)
        assert client.send(flood)["type"] == "result"
        assert client.shut() == 0
    finally:
        os.close(unread)


def test_a_clean_session_writes_nothing_to_stderr(
    connect: Callable[..., Client],
) -> None:
    client = connect(SKB_ARROW)
    client.send(INIT)
    assert client.send(call("echo", client.put("a", TABLE)))["type"] == "result"
    assert client.shut() == 0
    assert client.stderr() == ""


def test_a_nonblocking_stderr_costs_pieces_not_the_writer(
    connect: Callable[..., Client],
) -> None:
    unread, stderr = os.pipe()
    os.set_blocking(stderr, False)  # shared with the host's, as some callers leave it
    client = connect(FIXTURE, timeout=10, stderr=stderr)
    os.close(stderr)
    client.send(INIT)
    # Unread, the pipe fills: the drainer's writes fail with EAGAIN.
    client.send(call("flood", client.put("a", TABLE), how="fd2", bytes=FLOOD))
    log = bytearray()

    def read() -> None:
        while chunk := os.read(unread, 1 << 16):
            log.extend(chunk)

    reader = threading.Thread(target=read)
    reader.start()
    seen, deadline = -1, time.monotonic() + 5
    while len(log) != seen and time.monotonic() < deadline:  # the backlog settles
        seen = len(log)
        time.sleep(0.3)
    client.send(call("flood", client.put("b", TABLE), how="print", bytes=1))
    assert client.shut() == 0
    reader.join(10)
    os.close(unread)
    assert log.endswith(b"last words\n")


def test_a_line_without_its_newline_still_shows(
    connect: Callable[..., Client],
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    client.send(call("mutters", client.put("a", TABLE), dies=False))
    deadline = time.monotonic() + 5
    while b"no newline" not in client.errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.errors.endswith(b"no newline")  # while the host still runs
    assert client.shut() == 0


def test_a_line_without_its_newline_outlives_the_host(
    connect: Callable[..., Client],
) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    client.write(call("mutters", client.put("a", TABLE), dies=True))
    assert client.shut() == -signal.SIGKILL
    assert client.stderr().endswith("no newline")


# Slow: 4 KiB per 10 ms, so the 1 MiB held takes the drainer seconds to write.
# Late: stalled past the drainer's 1 s before the host exits, read only after.
@pytest.mark.parametrize(
    ("stall", "pause"),
    [(0, 0), (0, 0.01), (1.5, 0)],
    ids=["fast reader", "slow reader", "late reader"],
)
def test_dropped_stderr_is_counted_and_the_last_words_kept(
    connect: Callable[..., Client], stall: float, pause: float
) -> None:
    unread, stderr = os.pipe()
    client = connect(FIXTURE, timeout=10, stderr=stderr)
    os.close(stderr)
    client.send(INIT)
    client.send(call("flood", client.put("a", TABLE), how="fd2", bytes=FLOOD))
    time.sleep(stall)
    assert client.shut() == 0
    log = read_to_eof(unread, within=10, size=4096, pause=pause)
    os.close(unread)
    dropped = sum(int(n) for n in DROPPED.findall(log))
    kept = DROPPED.sub(b"", log)
    written = b"import-time print\nimport-time write\n" + b"x" * FLOOD + b"last words\n"
    assert dropped > 0
    assert len(kept) + dropped == len(written)
    assert kept.endswith(b"last words\n")


def test_a_stalled_stderr_cannot_keep_the_drainer_alive(
    connect: Callable[..., Client],
) -> None:
    unread, stderr = os.pipe()
    client = connect(FIXTURE, timeout=10, stderr=stderr)
    os.close(stderr)
    client.send(INIT)
    client.send(call("flood", client.put("a", TABLE), how="fd2", bytes=FLOOD))
    assert client.shut() == 0
    assert emptied(client.process.pid, within=3)
    os.close(unread)


# `lingers` execs a child that keeps the drainer alive; `forks` forks one, bare.
@pytest.mark.parametrize("child", ["lingers", "forks"])
def test_no_child_holds_the_channel(connect: Callable[..., Client], child: str) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    client.send(call(child, client.put("a", TABLE)))
    client.process.kill()
    client.process.wait()
    assert client.process.stdout
    start = time.monotonic()
    assert client.process.stdout.read() == b""
    assert time.monotonic() - start < 2  # not the 5 s the child lives
    assert client.process.stdin
    with pytest.raises(BrokenPipeError):  # unbuffered, so teardown has nothing to flush
        os.write(client.process.stdin.fileno(), b"\n")


def test_the_drainer_holds_no_fd_leaked_into_the_host(
    connect: Callable[..., Client],
) -> None:
    # As a caller leaking its end of the channel would, or any pipe it awaits EOF on.
    awaited, leaked = os.pipe()
    client = connect(FIXTURE, pass_fds=[leaked])
    os.close(leaked)
    client.send(INIT)
    client.send(call("lingers", client.put("a", TABLE)))
    client.process.kill()
    client.process.wait()
    assert read_to_eof(awaited, within=2) == b""  # not the 5 s the child lingers
    os.close(awaited)


def test_the_host_never_parents_the_drainer(connect: Callable[..., Client]) -> None:
    client = connect(FIXTURE)
    client.send(INIT)
    assert client.send(call("childless", client.put("a", TABLE)))["type"] == "result"
    assert client.shut() == 0


def test_a_failed_drainer_says_so(connect: Callable[..., Client]) -> None:
    client = connect(FIXTURE, env=os.environ | {"FIXTURE_BREAK_DRAINER": "1"})
    expected = "skb-arrow: stderr drainer failed: RuntimeError('broken on purpose')\n"
    # EOF when the drainer exits: the host holds the relay, not the caller's stderr.
    assert expected in client.stderr()


def test_sigint_to_the_process_group_leaves_the_last_words_relayed(
    connect: Callable[..., Client],
) -> None:
    client = connect(FIXTURE, process_group=0)
    client.send(INIT)
    os.killpg(client.process.pid, signal.SIGINT)
    client.process.wait(timeout=10)
    assert client.stderr().endswith("\nKeyboardInterrupt\n")  # the host's traceback


def test_stderr_into_the_channel_is_discarded(connect: Callable[..., Client]) -> None:
    client = connect(FIXTURE, stderr=subprocess.STDOUT)
    assert client.send(INIT)["type"] == "ready"
    assert client.send(call("noisy", client.put("a", TABLE)))["type"] == "result"
    assert client.shut() == 0


def test_sockets_without_an_inode_are_never_the_same_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As macOS reports every TCP socket: device 0, inode 0.
    tcp = os.stat_result((stat.S_IFSOCK | 0o777, 0, 0, 1, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(os, "fstat", lambda fd: tcp)
    assert not host._same_pipe(1, 2)


def test_stderr_into_a_socket_channel_is_discarded(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    ours, theirs = socket.socketpair()
    with theirs:
        process = subprocess.Popen(
            [*FIXTURE, "--segment-dir", str(directory)],
            stdin=subprocess.PIPE,
            stdout=theirs.fileno(),
            stderr=subprocess.STDOUT,
        )
    assert process.stdin
    process.stdin.write(b'{"type": "init", "protocol_version": 1}\n')
    process.stdin.close()
    with ours, ours.makefile("rb") as responses:
        lines = responses.read().splitlines()
    assert process.wait(timeout=30) == 0
    assert [json.loads(line)["type"] for line in lines] == ["ready"]


@pytest.mark.parametrize(
    "how",
    ["fd2", "lines", "halves"],
    ids=["one long line", "short lines", "a line in two writes"],
)
def test_stderr_goes_out_in_whole_lines_within_pipe_buf(
    connect: Callable[..., Client], how: str
) -> None:
    # A datagram per write shows where writes end, as a shared pipe would cut them.
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    messages: list[bytes] = []
    size = -(-3 * select.PIPE_BUF // 100) * 100  # 100-byte lines past PIPE_BUF
    flood = {
        "fd2": b"x" * size,
        "lines": (b"y" * 99 + b"\n") * (size // 100),
        "halves": b"z" * (size - 1) + b"\n",  # one line, written in two
    }[how]
    written = b"import-time print\nimport-time write\n" + flood + b"last words\n"

    def receive() -> None:
        ours.settimeout(10)
        while sum(map(len, messages)) < len(written):
            messages.append(ours.recv(1 << 16))

    reader = threading.Thread(target=receive)
    reader.start()
    # Held 1 s, not 50 ms: a CI runner can stall longer than the 10 ms between halves.
    env = os.environ | {"FIXTURE_PARTIAL": "1"} if how == "halves" else None
    client = connect(FIXTURE, stderr=theirs.fileno(), env=env)
    theirs.close()
    client.send(INIT)
    client.send(call("flood", client.put("a", TABLE), how=how, bytes=size))
    assert client.shut() == 0
    reader.join()
    ours.close()
    assert b"".join(messages) == written
    # A line is cut only when it alone is longer than PIPE_BUF.
    whole = [
        m.endswith(b"\n") or (len(m) == select.PIPE_BUF and b"\n" not in m)
        for m in messages
    ]
    assert all(len(m) <= select.PIPE_BUF for m in messages)
    assert all(whole)


def test_a_long_line_goes_out_as_it_grows(connect: Callable[..., Client]) -> None:
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    landed: list[float] = []  # when each piece of the line arrived

    def receive() -> None:
        ours.settimeout(10)
        while not (message := ours.recv(1 << 16)).endswith(b"last words\n"):
            if message.startswith(b"x"):
                landed.append(time.monotonic())

    reader = threading.Thread(target=receive)
    reader.start()
    client = connect(FIXTURE, stderr=theirs.fileno())
    theirs.close()
    client.send(INIT)
    flood = call(
        "flood", client.put("a", TABLE), how="trickle", bytes=3 * select.PIPE_BUF
    )
    client.send(flood)
    answered = time.monotonic()
    assert client.shut() == 0
    reader.join()
    ours.close()
    # PIPE_BUF bytes in after 0.2 s of the 0.6 s the line takes to write.
    assert landed and landed[0] < answered


def test_stderr_sharing_a_file_with_the_channel_is_kept(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    with open(tmp_path / "log", "w+b") as log:
        process = subprocess.Popen(
            [*FIXTURE, "--segment-dir", str(directory)],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process.communicate(b'{"type": "init", "protocol_version": 1}\n', timeout=30)
        assert process.returncode == 0
        assert emptied(process.pid, within=5)  # the drainer has written what it held
        log.seek(0)
        text = log.read()
    assert b'"type": "ready"' in text
    assert b"import-time print" in text


# Each way DIR can be unusable: (how it's made, the problem reported).
UNUSABLE = {
    "missing": "does not exist",
    "a file": "is not a directory",
    "mode 300": "is not readable",
    "mode 600": "is not readable",  # no search permission
    "mode 500": "is not writable",
    "non-empty": "is not empty",
}


@pytest.fixture
def unusable(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """DIR made unusable as the test's `how` param says."""
    how, path = request.getfixturevalue("how"), tmp_path / "dir"
    if how.startswith("mode") and os.geteuid() == 0:
        pytest.skip("root reads and writes anywhere")
    if how == "a file":
        path.write_bytes(b"")
    elif how != "missing":
        path.mkdir()
    if how == "non-empty":
        (path / "keep").write_bytes(b"x")
    if how.startswith("mode"):
        path.chmod(int(how.removeprefix("mode "), 8))
    yield path
    if path.is_dir():
        path.chmod(0o700)  # even when the test fails, so pytest can remove it


@pytest.mark.parametrize("how", UNUSABLE.keys())
def test_an_unusable_segment_dir_exits_2_untouched(unusable: Path, how: str) -> None:
    result = subprocess.run(
        [*SKB_ARROW, "--segment-dir", str(unusable)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode, result.stdout) == (2, b"")
    expected = f"skb-arrow: --segment-dir {unusable}: {UNUSABLE[how]}\n"
    assert result.stderr.decode() == expected
    if how == "non-empty":
        assert (unusable / "keep").read_bytes() == b"x"
