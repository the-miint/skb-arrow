"""Reference caller: runs a host, speaks the protocol, cleans up after it."""

import json
import os
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyarrow import ipc

from skb_arrow import transport

FIXTURE = [sys.executable, str(Path(__file__).with_name("fixture_host.py"))]
SKB_ARROW = [str(Path(sys.executable).parent / "skb-arrow")]
INIT = {"type": "init", "protocol_version": 1}


class Client:
    def __init__(
        self, command: list[str], directory: Path, timeout: float = 30, **popen: Any
    ) -> None:
        self.directory = directory
        # Read to EOF, as a caller should; a test passing `stderr` leaves it unread.
        popen.setdefault("stderr", subprocess.PIPE)
        # DIR relative to the host's cwd, as a caller may pass it.
        self.process = subprocess.Popen(
            [*command, "--segment-dir", directory.name],
            cwd=directory.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            **popen,
        )
        self.errors = bytearray()
        self.drain = threading.Thread(target=self._drain, daemon=True)
        if self.process.stderr:
            self.drain.start()
        # A wedged host fails its test instead of hanging it.
        self.watchdog = threading.Timer(timeout, self.process.kill)
        self.watchdog.daemon = True
        self.watchdog.start()

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
        status = self.process.wait()
        self.watchdog.cancel()
        return status

    def kill(self) -> None:
        """Stop the host if it still runs: test teardown."""
        self.watchdog.cancel()
        self.process.kill()
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
