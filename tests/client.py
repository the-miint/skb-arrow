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
        self, command: list[str], directory: Path, env: dict[str, str] | None = None
    ) -> None:
        self.directory = directory
        self.log = directory.with_name(f"{directory.name}.stderr")
        # stderr drains to a file: an undrained pipe can wedge the host.
        with self.log.open("wb") as stderr:
            self.process = subprocess.Popen(
                [*command, "--segment-dir", str(directory)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=env,
            )
        # A wedged host fails its test instead of hanging it.
        self.watchdog = threading.Timer(30, self.process.kill)
        self.watchdog.start()

    def write(self, message: object) -> None:
        assert self.process.stdin
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        self.process.stdin.flush()

    def receive(self) -> dict[str, Any]:
        assert self.process.stdout
        response = json.loads(self.process.stdout.readline())
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

    def clean(self) -> None:
        """Unlink the directory's entries, then remove it; what's gone is done."""
        with suppress(FileNotFoundError):
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    with suppress(FileNotFoundError):
                        os.unlink(entry.path)
            os.rmdir(self.directory)

    def stderr(self) -> str:
        return self.log.read_text()
