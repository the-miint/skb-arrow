"""One host through many calls: nothing leaks, nothing drifts (DESIGN §5 M4)."""

import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from client import FIXTURE, INIT, Client, emptied
from pyarrow import ipc
from test_capabilities import EXAMPLES

ROUNDS = 20
ECHO = {"table": pa.table({"n": np.arange(1 << 17)})}  # 1 MiB of int64


def distances(seed: int, n: int = 30) -> pa.Table:
    """Each pair of `n` random points once, with how far apart they are."""
    points = np.random.default_rng(seed).random((n, 2))
    a, b = np.triu_indices(n, 1)
    return pa.table(
        {"id_a": a, "id_b": b, "distance": np.hypot(*(points[a] - points[b]).T)}
    )


MANTEL = {"x": distances(1), "y": distances(2)}
CONSTANT = MANTEL | {"x": distances(1).set_column(2, "distance", pa.repeat(1.0, 435))}
# A round, in order. CONSTANT warns, and answers NaN.
CALLS: list[tuple[str, Mapping[str, pa.Table], dict[str, Any]]] = [
    ("echo", ECHO, {}),
    ("ancombc", *EXAMPLES["ancombc"]),
    ("mantel", MANTEL, {}),
    ("mantel", CONSTANT, {}),
    ("mantel", MANTEL | {"x": distances(1).slice(1)}, {}),  # a pair missing
    ("mantel", MANTEL, {"method": "nope"}),
]


@pytest.fixture
def host(tmp_path: Path) -> Iterator[Callable[[str], Client]]:
    """Start ready fixture hosts, each with a DIR and TMPDIR; any left are killed."""
    clients: list[Client] = []

    def start(name: str) -> Client:
        directory, private = tmp_path / name / "session", tmp_path / name / "tmp"
        directory.mkdir(parents=True)
        private.mkdir()
        environment = os.environ | {"TMPDIR": str(private)}
        clients.append(Client(FIXTURE, directory, timeout=120, env=environment))
        assert clients[-1].send(INIT | {"segment_bytes": 64 << 10})["type"] == "ready"
        return clients[-1]

    yield start
    for client in clients:
        client.kill()


def answer(client: Client, response: dict[str, Any]) -> tuple[Any, ...]:
    """What `response` says, an output as the bytes written (NaN equals NaN there)."""
    if response["type"] == "result":
        names = response["output"]
        said: tuple[Any, ...] = ([(client.directory / n).read_bytes() for n in names],)
        client.fetch(names)
    else:
        said = (response["kind"], response["message"])
    assert list(client.directory.iterdir()) == []
    return (response["type"], *said, response["warnings"])


def round_of_calls(client: Client) -> list[tuple[Any, ...]]:
    return [
        answer(client, client.call(name, tables, params, parts=4))
        for name, tables, params in CALLS
    ]


def decoded(segments: list[bytes]) -> pa.Table:
    return pa.Table.from_batches([b for s in segments for b in ipc.open_stream(s)])


def probe(client: Client) -> dict[str, Any]:
    response = client.send({"type": "call", "capability": "probe", "input": {}})
    found: dict[str, Any] = client.fetch(response["output"]).to_pylist()[0]
    return found


def resident(pid: int) -> int:
    """Bytes `pid` holds in memory now: ps reports KiB on both platforms."""
    ps = ["ps", "-o", "rss=", "-p", str(pid)]
    return int(subprocess.run(ps, capture_output=True, check=True).stdout) << 10


def test_a_soaked_host_leaks_nothing_and_answers_alike(
    host: Callable[[str], Client],
) -> None:
    client = host("soak")
    first = round_of_calls(client)
    assert [said[0] for said in first] == ["result"] * 4 + ["error"] * 2
    assert decoded(first[0][1]).equals(ECHO["table"])
    assert [w["category"] for w in first[3][-1]] == [
        "scipy.stats._warnings_errors.ConstantInputWarning"
    ]
    assert [said[1] for said in first[4:]] == ["invalid_input", "invalid_param"]
    probes, sizes = [probe(client)], [resident(client.process.pid)]
    for _ in range(ROUNDS - 1):
        assert round_of_calls(client) == first
        probes.append(probe(client))
        sizes.append(resident(client.process.pid))
    # fds, threads, children and Arrow's bytes, once the first round has set up.
    assert all(p == probes[1] for p in probes[1:])
    assert sizes[-1] - sizes[1] < 8 << 20
    # An output left unread is cleaned with DIR.
    assert client.call("echo", ECHO)["type"] == "result"
    assert any(client.directory.iterdir())
    assert client.shut() == 0
    # Before stderr is read: that waits for the drainer to let go of it.
    assert emptied(client.process.pid, 5)
    assert not client.directory.exists()
    assert list(client.directory.with_name("tmp").iterdir()) == []
    assert "Traceback" not in client.stderr()


def test_a_warm_host_writes_what_a_fresh_one_does(
    host: Callable[[str], Client],
) -> None:
    fresh, warm = host("fresh"), host("warm")
    for _ in range(4):  # 24 mixed calls
        round_of_calls(warm)
    assert round_of_calls(warm) == round_of_calls(fresh)
