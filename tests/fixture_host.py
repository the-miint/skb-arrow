"""A host with test capabilities: fixture_host.py --segment-dir DIR."""

import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from skb_arrow import host

if TYPE_CHECKING:
    import pyarrow as pa


def noisy(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
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


def dies(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    print("about to die")
    return tables["table"]


def main() -> int:
    channel = host.reserve()
    # As a capability's library might, on import.
    print("import-time print")
    os.write(1, b"import-time write\n")

    from skb_arrow import protocol, registry

    table = frozenset({"table"})
    registry.CAPABILITIES |= {
        "noisy": registry.Capability(1, table, frozenset(), noisy),
        "fail": registry.Capability(1, table, frozenset({"kind"}), fail),
        "dies": registry.Capability(1, table, frozenset(), dies),
    }
    handle = protocol.handle

    def killed_before_replying(session: protocol.Session, line: bytes) -> bytes:
        response = handle(session, line)
        if json.loads(line).get("capability") == "dies":
            os.kill(os.getpid(), signal.SIGKILL)
        return response

    protocol.handle = killed_before_replying
    return host.serve(Path(sys.argv[-1]), *channel)


if __name__ == "__main__":
    sys.exit(main())
