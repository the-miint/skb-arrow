"""Reference caller: runs a host, speaks the protocol, cleans up after it."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyarrow import ipc

from skb_arrow import transport

FIXTURE = [sys.executable, str(Path(__file__).with_name("fixture_host.py"))]
SKB_ARROW = [str(Path(sys.executable).parent / "skb-arrow")]
INIT = {"type": "init", "protocol_version": 1}
# Where a caller places DIR (docs/transport.md#session-directory).
PLACE = "/dev/shm" if sys.platform == "linux" else tempfile.gettempdir()


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


class Client:
    """A host leading its own process group: pgid == pid."""

    def __init__(
        self, command: list[str], directory: Path, timeout: float = 30, **popen: Any
    ) -> None:
        self.directory = directory
        # Read to EOF, as a caller should; a test passing `stderr` leaves it unread.
        popen.setdefault("stderr", subprocess.PIPE)
        # So stopping it stops all it started: one that outlives it holding the channel
        # would hang the test. (setsid, then setpgid, fails.)
        if "process_group" not in popen:
            popen.setdefault("start_new_session", True)
        popen.setdefault("stdout", subprocess.PIPE)
        # DIR relative to the host's cwd, as a caller may pass it.
        self.process = subprocess.Popen(
            [*command, "--segment-dir", directory.name],
            cwd=directory.parent,
            stdin=subprocess.PIPE,
            **popen,
        )
        self.errors = bytearray()
        self.peak_rss = 0  # bytes, once shut
        self.drain = threading.Thread(target=self._drain, daemon=True)
        if self.process.stderr:
            self.drain.start()
        # A wedged host fails its test instead of hanging it.
        self.watchdog = threading.Timer(timeout, self._stop)
        self.watchdog.daemon = True
        self.watchdog.start()

    def _stop(self) -> None:
        # Gone, or on macOS holding only zombies (EPERM).
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGKILL)

    def _drain(self) -> None:
        assert self.process.stderr
        # Teed beside DIR as it comes: a trail for a test that fails.
        log = self.directory.with_name(f"{self.directory.name}.stderr")
        with log.open("wb") as file:
            while chunk := os.read(self.process.stderr.fileno(), 1 << 16):
                file.write(chunk)
                file.flush()
                self.errors += chunk

    def write(self, message: object) -> None:
        assert self.process.stdin
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        self.process.stdin.flush()

    def receive(self) -> dict[str, Any]:
        assert self.process.stdout
        line = self.process.stdout.readline()
        assert line, "the host closed its responses"
        response = json.loads(line)
        assert type(response) is dict
        return response

    def send(self, message: object) -> dict[str, Any]:
        self.write(message)
        return self.receive()

    def put(self, name: str, data: pa.Table | bytes) -> str:
        if isinstance(data, bytes):
            (self.directory / name).write_bytes(data)
            return name
        path = self.directory / name
        with open(path, "xb") as file, ipc.new_stream(file, data.schema) as writer:
            writer.write_table(data)
        return name

    def fetch(self, names: list[str]) -> pa.Table:
        return transport.read(self.directory, names)

    def shut(self) -> int:
        """Close both pipes and wait for the host: its exit status."""
        assert self.process.stdin and self.process.stdout
        self.process.stdin.close()
        self.process.stdout.close()
        _, status, usage = os.wait4(self.process.pid, 0)
        self.process.returncode = os.waitstatus_to_exitcode(status)
        # macOS reports bytes, Linux KiB.
        self.peak_rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.watchdog.cancel()
        return self.process.returncode

    def kill(self) -> None:
        """Stop the host and all it started, if they still run: test teardown."""
        self.watchdog.cancel()
        self._stop()
        self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe:
                pipe.close()

    def clean(self) -> None:
        """Unlink the directory's entries, then remove it; what's gone is done."""
        with suppress(FileNotFoundError):
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    with suppress(FileNotFoundError):
                        os.unlink(entry.path)
            os.rmdir(self.directory)

    def stderr(self) -> str:
        """All the host wrote to stderr: waits for its EOF."""
        assert self.process.stderr, "stderr is the test's, not a drained pipe"
        self.drain.join(30)
        assert not self.drain.is_alive(), "stderr never reached EOF"
        return self.errors.decode()
