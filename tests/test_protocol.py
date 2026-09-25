import json
import warnings
from collections.abc import Callable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pyarrow import ipc

from skb_arrow import protocol, registry, transport
from skb_arrow.errors import InvalidInput

INIT = {"type": "init", "protocol_version": 1}
TABLE = pa.table({"a": pa.array(range(1000), pa.int64())})
Run = Callable[[Mapping[str, pa.Table], Mapping[str, object]], pa.Table]


def receive(line: bytes) -> dict[str, Any]:
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    response = json.loads(line)
    assert type(response) is dict
    return response


def send(session: protocol.Session, message: object) -> dict[str, Any]:
    return receive(protocol.handle(session, json.dumps(message).encode() + b"\n"))


def call(inputs: object, capability: object = "echo", **fields: object) -> object:
    return {"type": "call", "capability": capability, "input": inputs, **fields}


def put(directory: Path, name: str, table: pa.Table = TABLE) -> str:
    with open(directory / name, "xb") as file, ipc.new_stream(file, table.schema) as w:
        w.write_table(table)
    return name


def register(
    monkeypatch: pytest.MonkeyPatch, name: str, run: Run, inputs: tuple[str, ...]
) -> None:
    capability = registry.Capability(1, frozenset(inputs), frozenset(), run)
    monkeypatch.setitem(registry.CAPABILITIES, name, capability)


def pair(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    return pa.table({"left": tables["left"]["a"], "right": tables["right"]["a"]})


def fail(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    raise InvalidInput("no")


@pytest.fixture
def session(tmp_path: Path) -> protocol.Session:
    session = protocol.Session(tmp_path)
    assert send(session, INIT)["type"] == "ready"
    return session


def test_init_replies_ready(tmp_path: Path) -> None:
    assert send(protocol.Session(tmp_path), INIT) == {
        "type": "ready",
        "protocol_version": 1,
        "host_version": version("skb-arrow"),
        "capabilities": {"echo": 1},
    }


def test_init_defaults_null_fields_and_ignores_unknown_ones(tmp_path: Path) -> None:
    session = protocol.Session(tmp_path)
    assert send(session, INIT | {"segment_bytes": None, "x": 1})["type"] == "ready"
    assert session.segment_bytes == 256 << 20


def test_another_protocol_version_is_answered_with_the_hosts(tmp_path: Path) -> None:
    assert send(protocol.Session(tmp_path), INIT | {"protocol_version": 2}) == {
        "type": "error",
        "kind": "host_incompatible",
        "message": "protocol_version 2; this host speaks 1",
        "protocol_version": 1,
        "id": None,
        "warnings": [],
    }


@pytest.mark.parametrize(
    "init",
    [
        INIT | {"protocol_version": True},
        {"type": "init"},
        INIT | {"segment_bytes": "1 MiB"},
        INIT | {"segment_bytes": 1023},
    ],
    ids=["bool version", "no version", "mistyped segment_bytes", "segment_bytes 1023"],
)
def test_a_bad_init_is_host_incompatible(tmp_path: Path, init: object) -> None:
    session = protocol.Session(tmp_path)
    response = send(session, init)
    assert (response["kind"], response.get("protocol_version")) == (
        "host_incompatible",
        1,
    )
    assert session.segment_bytes is None


def test_segment_bytes_may_be_1024(tmp_path: Path) -> None:
    session = protocol.Session(tmp_path)
    assert send(session, INIT | {"segment_bytes": 1024})["type"] == "ready"
    assert session.segment_bytes == 1024


def test_init_twice_is_host_incompatible(session: protocol.Session) -> None:
    response = send(session, INIT)
    assert (response["kind"], response["message"]) == (
        "host_incompatible",
        "init sent twice",
    )


@pytest.mark.parametrize(
    "line",
    [
        b"\xff\n",
        b"{\n",
        b"[1]\n",
        b"\n",
        b"[" * 100_000 + b"\n",
        b'{"type": "init", "protocol_version": ' + b"1" * 5000 + b"}\n",
    ],
    ids=["not utf-8", "bad json", "not an object", "empty", "deep", "huge int"],
)
def test_an_unreadable_line_is_host_incompatible(
    session: protocol.Session, line: bytes
) -> None:
    response = receive(protocol.handle(session, line))
    assert (response["type"], response["kind"], response["id"]) == (
        "error",
        "host_incompatible",
        None,
    )


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ({"type": "shutdown"}, "unknown message type 'shutdown'"),
        ({"capability": "echo"}, "missing field 'type'"),
        ({"type": 1}, "field 'type' must be a string"),
    ],
    ids=["unknown", "missing", "mistyped"],
)
def test_a_bad_type_is_host_incompatible(
    session: protocol.Session, message: object, reason: str
) -> None:
    response = send(session, message)
    assert (response["kind"], response["message"]) == ("host_incompatible", reason)


def test_a_call_before_init_is_rejected_and_its_segments_disposed(
    tmp_path: Path,
) -> None:
    response = send(protocol.Session(tmp_path), call({"table": [put(tmp_path, "a")]}))
    assert (response["kind"], response["message"]) == (
        "host_incompatible",
        "call before init",
    )
    assert list(tmp_path.iterdir()) == []


def test_echo_round_trips_a_table_across_segments(tmp_path: Path) -> None:
    session = protocol.Session(tmp_path)
    send(session, INIT | {"segment_bytes": 1024})
    names = [put(tmp_path, f"in-{i}", TABLE.slice(i * 250, 250)) for i in range(4)]
    response = send(session, call({"table": names}, id=1))
    assert (response["type"], response["id"], response.get("warnings")) == (
        "result",
        1,
        [],
    )
    # 8000 bytes under a 1024 cap: eight 125-row segments.
    assert len(response["output"]) == 8
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(response["output"])
    assert transport.read(tmp_path, response["output"]).equals(TABLE)


def test_output_names_continue_across_calls(session: protocol.Session) -> None:
    outputs = [
        send(session, call({"table": [put(session.directory, "a")]}))["output"]
        for _ in range(2)
    ]
    assert outputs == [["skbout-0"], ["skbout-1"]]


def test_named_tables_reach_the_capability_by_name(
    session: protocol.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    register(monkeypatch, "pair", pair, ("left", "right"))
    left = put(session.directory, "l", pa.table({"a": [1, 2]}))
    right = put(session.directory, "r", pa.table({"a": [3, 4]}))
    response = send(session, call({"right": [right], "left": [left]}, "pair"))
    output = transport.read(session.directory, response["output"])
    assert output.to_pydict() == {"left": [1, 2], "right": [3, 4]}


@pytest.mark.parametrize("ident", ["x", 7])
def test_the_id_is_echoed_on_results_and_errors(
    session: protocol.Session, ident: str | int
) -> None:
    ok = send(session, call({"table": [put(session.directory, "a")]}, id=ident))
    bad = send(session, call({"table": ["b"]}, "nope", id=ident))
    assert (ok["type"], ok["id"]) == ("result", ident)
    assert (bad["type"], bad["id"]) == ("error", ident)


@pytest.mark.parametrize("ident", [True, [1], 1.5], ids=repr)
def test_a_mistyped_id_is_answered_with_a_null_id(
    session: protocol.Session, ident: object
) -> None:
    response = send(session, call({"table": [put(session.directory, "a")]}, id=ident))
    assert (response["kind"], response["id"]) == ("host_incompatible", None)


def test_null_fields_are_absent_and_unknown_fields_ignored(
    session: protocol.Session,
) -> None:
    message = call(
        {"table": [put(session.directory, "a")]}, id=None, params=None, x={"y": 1}
    )
    response = send(session, message)
    assert (response["type"], response["id"]) == ("result", None)


def test_an_error_envelope_holds_exactly_its_fields(session: protocol.Session) -> None:
    assert send(session, call({"table": ["a"]}, "nope", id="q")) == {
        "type": "error",
        "kind": "host_incompatible",
        "message": "unknown capability 'nope'",
        "id": "q",
        "warnings": [],
    }


REJECTED = {
    "unknown capability": (call({"table": ["a", "b"]}, "nope"), "host_incompatible"),
    "mistyped capability": (call({"table": ["a", "b"]}, 5), "host_incompatible"),
    "mistyped params": (call({"table": ["a", "b"]}, params=[1]), "host_incompatible"),
    "bool id": (call({"table": ["a", "b"]}, id=True), "host_incompatible"),
    "list id": (call({"table": ["a", "b"]}, id=[1]), "host_incompatible"),
    "unknown param": (call({"table": ["a", "b"]}, params={"seed": 1}), "invalid_param"),
    "other inputs": (call({"left": ["a"], "table": ["b"]}), "invalid_param"),
    "bad name": (call({"table": ["a", "b", "../c"]}), "host_incompatible"),
    "non-string name": (call({"table": ["a", "b", 5]}), "host_incompatible"),
    "name twice": (
        call({"left": ["a"], "right": ["b", "A"]}, "pair"),
        "host_incompatible",
    ),
    "empty table": (
        call({"left": ["a", "b"], "right": []}, "pair"),
        "host_incompatible",
    ),
    "failing run": (call({"table": ["a", "b"]}, "fail"), "invalid_input"),
}


@pytest.mark.parametrize(("message", "kind"), REJECTED.values(), ids=REJECTED.keys())
def test_a_rejected_call_disposes_every_segment_it_names(
    session: protocol.Session,
    monkeypatch: pytest.MonkeyPatch,
    message: object,
    kind: str,
) -> None:
    register(monkeypatch, "pair", pair, ("left", "right"))
    register(monkeypatch, "fail", fail, ("table",))
    for name in ["a", "b"]:
        put(session.directory, name)
    assert send(session, message)["kind"] == kind
    assert list(session.directory.iterdir()) == []


@pytest.mark.parametrize(
    "inputs", [None, "a", ["a"], {"table": "a"}, {"table": [["a"]]}], ids=repr
)
def test_a_mistyped_input_is_host_incompatible(
    session: protocol.Session, inputs: object
) -> None:
    assert send(session, call(inputs))["kind"] == "host_incompatible"


def test_a_corrupt_segment_is_invalid_input(session: protocol.Session) -> None:
    (session.directory / "a").write_bytes(b"not arrow")
    response = send(session, call({"table": ["a"]}))
    assert (response["kind"], response["message"][:11]) == (
        "invalid_input",
        "segment a: ",
    )
    assert list(session.directory.iterdir()) == []


def test_a_valueerror_is_the_callers_from_run_but_a_bug_from_machinery(
    session: protocol.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raises(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> None:
        raise ValueError("bad value")

    def broken(*args: object) -> None:
        raise ValueError("bug")

    register(monkeypatch, "raises", raises, ("table",))
    from_run = send(session, call({"table": [put(session.directory, "a")]}, "raises"))
    monkeypatch.setattr(transport, "write", broken)
    from_write = send(session, call({"table": [put(session.directory, "b")]}))
    assert (from_run["kind"], "traceback" in from_run) == ("invalid_input", False)
    assert (from_write["kind"], "traceback" in from_write) == ("internal", True)


@pytest.mark.parametrize("fails", [False, True])
def test_warnings_ride_on_results_and_errors(
    session: protocol.Session, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    def run(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
        warnings.warn("careful", UserWarning, stacklevel=1)
        if fails:
            raise InvalidInput("no")
        return tables["table"]

    register(monkeypatch, "warns", run, ("table",))
    response = send(session, call({"table": [put(session.directory, "a")]}, "warns"))
    assert response["type"] == ("error" if fails else "result")
    assert response["warnings"] == [
        {"category": "UserWarning", "message": "careful", "count": 1}
    ]


def test_warnings_cover_reading_and_writing(
    session: protocol.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    read, write = transport.read, transport.write

    def warn_then(message: str, step: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any) -> Any:
            warnings.warn(message, UserWarning, stacklevel=1)
            return step(*args)

        return wrapped

    monkeypatch.setattr(transport, "read", warn_then("reading", read))
    monkeypatch.setattr(transport, "write", warn_then("writing", write))
    response = send(session, call({"table": [put(session.directory, "a")]}))
    assert [w["message"] for w in response["warnings"]] == ["reading", "writing"]
