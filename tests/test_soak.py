"""One host through many calls: nothing leaks, nothing drifts (DESIGN §5 M4)."""

import os
import signal
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from client import FIXTURE, INIT, Client, emptied

ROUNDS = 20
ECHO = {"table": pa.table({"n": np.arange(1 << 17)})}  # 1 MiB of int64
_SAMPLES = [f"s{i}" for i in range(12)]
_COUNTS = np.random.default_rng(0).integers(1, 50, size=(12, 5))
ANCOMBC = {
    "table": pa.table(
        {
            "sample_id": np.repeat(_SAMPLES, 5).tolist(),
            "feature_id": [f"f{j}" for j in range(5)] * 12,
            "value": _COUNTS.ravel().tolist(),
        }
    ),
    "metadata": pa.table({"sample_id": _SAMPLES, "g": ["a", "b", "c"] * 4}),
}
DUNNETT: dict[str, Any] = {
    "formula": "g",
    "grouping": "g",
    "posthoc": ["dunnett"],
    "bootstraps": 10,
}


def distances(seed: int, n: int = 30) -> pa.Table:
    """Each pair of `n` random points once, with how far apart they are."""
    points = np.random.default_rng(seed).random((n, 2))
    a, b = np.triu_indices(n, 1)
    return pa.table(
        {"id_a": a, "id_b": b, "distance": np.hypot(*(points[a] - points[b]).T)}
    )


MANTEL = {"x": distances(1), "y": distances(2)}
MISSING_A_PAIR = {"x": distances(1).slice(1), "y": distances(2)}


@pytest.fixture
def host(tmp_path: Path) -> Iterator[Callable[[str], Client]]:
    """Start ready fixture hosts, each with a DIR and TMPDIR; any left are killed."""
    clients: list[Client] = []

    def start(name: str) -> Client:
        directory, private = tmp_path / name / "session", tmp_path / name / "tmp"
        directory.mkdir(parents=True)
        private.mkdir()
        clients.append(
            Client(FIXTURE, directory, env=os.environ | {"TMPDIR": str(private)})
        )
        assert clients[-1].send(INIT | {"segment_bytes": 64 << 10})["type"] == "ready"
        return clients[-1]

    yield start
    for client in clients:
        client.kill()


def call(
    client: Client,
    capability: str,
    tables: Mapping[str, pa.Table],
    parts: int = 1,
    **params: object,
) -> dict[str, Any]:
    """`capability` on `tables`, each put in `parts` segments."""
    inputs = {}
    for name, table in tables.items():
        rows = -(-table.num_rows // parts)
        inputs[name] = [
            client.put(f"{name}-{i}", table.slice(i * rows, rows)) for i in range(parts)
        ]
    message = {"type": "call", "capability": capability, "params": params}
    return client.send(message | {"input": inputs})


def answer(client: Client, response: dict[str, Any]) -> tuple[object, ...]:
    """What `response` says, its output fetched, once DIR is empty again."""
    if response["type"] == "result":
        said: tuple[object, ...] = (client.fetch(response["output"]),)
    else:
        said = (response["kind"], response["message"])
    assert list(client.directory.iterdir()) == []
    return (*said, response["warnings"])


def round_of_calls(client: Client) -> list[tuple[object, ...]]:
    return [
        answer(client, call(client, "echo", ECHO, parts=4)),
        answer(client, call(client, "ancombc", ANCOMBC, **DUNNETT)),
        answer(client, call(client, "mantel", MANTEL)),
        answer(client, call(client, "mantel", MISSING_A_PAIR)),
        answer(client, call(client, "mantel", MANTEL, method="nope")),
    ]


def probe(client: Client) -> dict[str, Any]:
    response = client.send({"type": "call", "capability": "probe", "input": {}})
    found: dict[str, Any] = client.fetch(response["output"]).to_pylist()[0]
    return found


def resident(pid: int) -> int:
    """Bytes `pid` holds in memory now: ps reports KiB on both platforms."""
    ps = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True)
    return int(ps.stdout) << 10


def test_a_soaked_host_leaks_nothing_and_answers_alike(
    host: Callable[[str], Client],
) -> None:
    client = host("soak")
    first = round_of_calls(client)
    assert [said[0] for said in first[3:]] == ["invalid_input", "invalid_param"]
    probes, sizes = [probe(client)], [resident(client.process.pid)]
    for _ in range(ROUNDS - 1):
        assert round_of_calls(client) == first
        probes.append(probe(client))
        sizes.append(resident(client.process.pid))
    assert all(p == probes[1] for p in probes[1:])  # fds, Arrow's bytes, numba's layer
    assert sizes[-1] - sizes[1] < 8 << 20
    # An output left unread is cleaned with DIR.
    assert call(client, "echo", ECHO)["type"] == "result"
    assert any(client.directory.iterdir())
    assert client.shut() == 0
    # Before stderr is read: that waits for the drainer to let go of it.
    assert emptied(client.process.pid, 5)
    assert not client.directory.exists()
    assert list(client.directory.with_name("tmp").iterdir()) == []
    assert "Traceback" not in client.stderr()


def test_a_host_killed_with_numbas_threads_running_leaves_no_process(
    host: Callable[[str], Client],
) -> None:
    client = host("killed")
    answer(client, call(client, "mantel", MANTEL))
    assert probe(client)["layer"] is not None  # numba's pool has started
    os.kill(client.process.pid, signal.SIGKILL)
    assert client.process.wait(timeout=30) == -signal.SIGKILL
    assert emptied(client.process.pid, 5)


def test_a_warm_host_writes_what_a_fresh_one_does(
    host: Callable[[str], Client],
) -> None:
    fresh, warm = host("fresh"), host("warm")
    for _ in range(4):  # 20 mixed calls
        round_of_calls(warm)

    def written(client: Client) -> list[tuple[list[bytes], object]]:
        found = []
        for response in (
            call(client, "ancombc", ANCOMBC, **DUNNETT),
            call(client, "mantel", MANTEL),
        ):
            names = response["output"]
            found.append(
                (
                    [(client.directory / n).read_bytes() for n in names],
                    response["warnings"],
                )
            )
            client.fetch(names)
        return found

    assert written(warm) == written(fresh)
