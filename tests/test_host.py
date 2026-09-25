import os
import signal
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest
from client import FIXTURE, INIT, SKB_ARROW, Client

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


def test_echo_round_trips_segments_and_shutdown_removes_dir(directory: Path) -> None:
    client = Client(SKB_ARROW, directory)
    assert client.send(INIT | {"segment_bytes": 1024})["capabilities"] == {"echo": 1}
    names = [client.put(f"in-{i}", TABLE.slice(i * 250, 250)) for i in range(4)]
    result = client.send(call("echo", "unused") | {"input": {"table": names}})
    assert len(result["output"]) == 8
    assert client.fetch(result["output"]).equals(TABLE)
    # Left for the host: an output never fetched, and a segment never named.
    assert client.send(call("echo", client.put("a", TABLE)))["type"] == "result"
    client.put("unnamed", TABLE)
    assert client.shut() == 0
    assert not directory.exists()


def test_every_kind_end_to_end(directory: Path) -> None:
    client = Client(FIXTURE, directory)
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


def test_the_channel_stays_clean_whatever_prints(directory: Path) -> None:
    client = Client(FIXTURE, directory)
    client.send(INIT)
    # Pipelined: the second request is already waiting while `noisy` reads stdin.
    client.write(call("noisy", client.put("a", TABLE)))
    client.write(call("echo", client.put("b", TABLE)))
    assert [client.receive()["type"] for _ in range(2)] == ["result", "result"]
    assert client.shut() == 0
    log = client.stderr()
    for noise in ["import-time print", "import-time write", "print during run"]:
        assert noise in log
    for noise in ["write during run", "child during run"]:
        assert noise in log


def test_a_killed_host_leaves_outputs_only_in_dir_for_the_caller_to_clean(
    tmp_path: Path, directory: Path
) -> None:
    private = tmp_path / "tmp"
    private.mkdir()
    client = Client(FIXTURE, directory, env=os.environ | {"TMPDIR": str(private)})
    client.send(INIT)
    client.write(call("dies", client.put("a", TABLE)))
    assert client.shut() == -signal.SIGKILL
    assert [p.name for p in directory.iterdir()] == ["skbout-0"]
    assert list(private.iterdir()) == []
    assert "about to die" in client.stderr()
    client.clean()
    assert not directory.exists()


def test_a_caller_that_stops_reading_is_cleaned_up_after(directory: Path) -> None:
    client = Client(FIXTURE, directory)
    client.send(INIT)
    assert client.process.stdout
    client.process.stdout.close()
    client.write(call("echo", client.put("a", TABLE)))
    assert client.process.wait(timeout=30) == 1
    assert not directory.exists()
    assert "Traceback" not in client.stderr()


def make_unusable(path: Path, problem: str) -> None:
    if problem == "is not a directory":
        path.write_bytes(b"")
    elif problem != "does not exist":
        path.mkdir()
    if problem == "is not empty":
        (path / "keep").write_bytes(b"x")
    if problem == "is not writable":
        path.chmod(0o500)


@pytest.mark.parametrize(
    "problem",
    ["does not exist", "is not a directory", "is not empty", "is not writable"],
)
def test_an_unusable_segment_dir_exits_2_untouched(
    tmp_path: Path, problem: str
) -> None:
    if problem == "is not writable" and os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    target = tmp_path / "dir"
    make_unusable(target, problem)
    result = subprocess.run(
        [*SKB_ARROW, "--segment-dir", str(target)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode, result.stdout) == (2, b"")
    assert result.stderr.decode() == f"skb-arrow: --segment-dir {target}: {problem}\n"
    if problem == "is not empty":
        assert (target / "keep").read_bytes() == b"x"
