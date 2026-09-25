import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyarrow as pa

from skb_arrow import registry, transport
from skb_arrow.errors import HostIncompatible, classify, collect_warnings

PROTOCOL_VERSION = 1
_DEFAULT_SEGMENT_BYTES = 256 << 20
_MIN_SEGMENT_BYTES = 1024
_ID_TYPES = (str, int)
_JSON_TYPES = {str: "a string", int: "an integer", dict: "an object", list: "an array"}
_REQUIRED = object()


@dataclass
class Session:
    directory: Path
    ready: bool = False  # an init succeeded
    segment_bytes: int = _DEFAULT_SEGMENT_BYTES
    outputs: Iterator[int] = field(default_factory=itertools.count)


def handle(session: Session, line: bytes) -> bytes:
    """The response line to one request line."""
    request: dict[str, Any] = {}
    with collect_warnings() as warnings:
        try:
            request = _parse(line)
            response = _respond(session, request)
        except Exception as e:
            response = {"type": "error", **classify(e, capability=False)}
            if not session.ready:
                response["protocol_version"] = PROTOCOL_VERSION
    ident = request.get("id")
    response["id"] = ident if type(ident) in _ID_TYPES else None
    response["warnings"] = warnings
    return (json.dumps(response) + "\n").encode()


def _parse(line: bytes) -> dict[str, Any]:
    try:
        request = json.loads(line.decode(), parse_constant=_not_json)
    except (ValueError, RecursionError) as e:
        raise HostIncompatible(f"unreadable request: {e}") from e
    if type(request) is not dict:
        raise HostIncompatible("a request must be a JSON object")
    return request


def _not_json(constant: str) -> None:
    raise ValueError(f"{constant} is not JSON")


def _respond(session: Session, request: dict[str, Any]) -> dict[str, Any]:
    try:
        kind = _field(request, "type", str)
        if kind == "init":
            return _init(session, request)
        if kind != "call":
            raise HostIncompatible(f"unknown message type {kind!r}")
        capability, tables, params = _prepare(session, request)
    finally:
        # Before the capability runs, so no output can exist if this fails.
        transport.dispose(session.directory, _named(request.get("input")))
    return _run(session, capability, tables, params)


def _init(session: Session, request: dict[str, Any]) -> dict[str, Any]:
    if session.ready:
        raise HostIncompatible("init sent twice")
    requested = _field(request, "protocol_version", int)
    if requested != PROTOCOL_VERSION:
        raise HostIncompatible(
            f"protocol_version {requested}; this host speaks {PROTOCOL_VERSION}"
        )
    segment_bytes = _field(
        request, "segment_bytes", int, default=_DEFAULT_SEGMENT_BYTES
    )
    if segment_bytes < _MIN_SEGMENT_BYTES:
        raise HostIncompatible(f"segment_bytes must be at least {_MIN_SEGMENT_BYTES}")
    session.ready, session.segment_bytes = True, segment_bytes
    return {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "host_version": version("skb-arrow"),
        "capabilities": registry.schema_versions(),
    }


def _prepare(
    session: Session, request: dict[str, Any]
) -> tuple[registry.Capability, dict[str, pa.Table], dict[str, Any]]:
    """A call's capability, input tables, and params, once all are checked."""
    if not session.ready:
        raise HostIncompatible("call before init")
    _field(request, "id", *_ID_TYPES, default=None)  # checked here; `handle` echoes it
    inputs = _field(request, "input", dict)
    for table, names in inputs.items():
        if (
            type(names) is not list
            or not names
            or any(type(n) is not str for n in names)
        ):
            raise HostIncompatible(
                f"input {table!r} must be a non-empty list of segment names"
            )
    transport.check_names(name for names in inputs.values() for name in names)
    params = _field(request, "params", dict, default={})
    capability = registry.validate(_field(request, "capability", str), inputs, params)
    tables = {t: transport.read(session.directory, s) for t, s in inputs.items()}
    return capability, tables, params


def _run(
    session: Session,
    capability: registry.Capability,
    tables: dict[str, pa.Table],
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        output = capability.run(tables, params)
    except Exception as e:
        return {"type": "error", **classify(e, capability=True)}
    names = transport.write(
        session.directory, output, session.segment_bytes, session.outputs
    )
    return {"type": "result", "output": names}


def _named(value: object) -> list[str]:
    """Every string among `value`'s values, at any depth: what a call names."""
    names, pending = [], [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            names.append(item)
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.values())
    return names


def _field(
    message: dict[str, Any], key: str, *types: type, default: object = _REQUIRED
) -> Any:
    """`message[key]`, of exactly one of `types`; null counts as absent."""
    value = message.get(key)
    if value is None:
        if default is _REQUIRED:
            raise HostIncompatible(f"missing field {key!r}")
        return default
    if type(value) not in types:
        expected = " or ".join(_JSON_TYPES[t] for t in types)
        raise HostIncompatible(f"field {key!r} must be {expected}")
    return value
