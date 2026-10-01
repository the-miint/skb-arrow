import os
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from client import FIXTURE, INIT, SKB_ARROW, Client

from skb_arrow import registry

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
