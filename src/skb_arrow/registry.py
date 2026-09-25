import sys
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from skb_arrow.capabilities import echo
from skb_arrow.errors import HostIncompatible, InvalidParam

_REQUIRED = object()
_TYPES = {str: "a string", int: "an integer", list: "an array"}


@dataclass(frozen=True)
class Param:
    type: type  # its JSON type; `float` is a number
    default: object = _REQUIRED
    valid: Callable[[Any], bool] | None = None
    rule: str = ""  # what `valid` requires: completes "must be …"


@dataclass(frozen=True)
class Capability:
    schema_version: int
    inputs: frozenset[str]
    params: Mapping[str, Param]
    run: Callable[[Mapping[str, pa.Table], Mapping[str, Any]], pa.Table]


CAPABILITIES = {"echo": Capability(1, frozenset({"table"}), {}, echo.run)}


def validate(
    name: str, inputs: Collection[str], params: Mapping[str, object]
) -> tuple[Capability, dict[str, Any]]:
    """Capability `name` and a call's params resolved, once the call suits it."""
    capability = CAPABILITIES.get(name)
    if capability is None:
        raise HostIncompatible(f"unknown capability {name!r}")
    if unknown := params.keys() - capability.params.keys():
        raise InvalidParam(f"unknown {name} params: {', '.join(sorted(unknown))}")
    if set(inputs) != capability.inputs:
        raise InvalidParam(
            f"{name} takes inputs: {', '.join(sorted(capability.inputs))}"
        )
    given = {key: value for key, value in params.items() if value is not None}
    return capability, {
        key: _resolve(f"{name} param {key!r}", param, given.get(key, _REQUIRED))
        for key, param in capability.params.items()
    }


def _resolve(what: str, param: Param, value: object) -> object:
    if value is _REQUIRED:
        if param.default is _REQUIRED:
            raise InvalidParam(f"{what} is required")
        return param.default
    if param.type is float:
        value = _number(what, value)
    elif type(value) is not param.type:
        raise InvalidParam(f"{what} must be {_TYPES[param.type]}")
    if param.valid is not None and not param.valid(value):
        raise InvalidParam(f"{what} must be {param.rule}")
    return value


def _number(what: str, value: object) -> float:
    """`value` as a finite float: any JSON number, integers included."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidParam(f"{what} must be a number")
    # Compared exactly, before float() could overflow on a huge integer.
    if not abs(value) <= sys.float_info.max:
        raise InvalidParam(f"{what} must be a finite number")
    return float(value)


def schema_versions() -> dict[str, int]:
    return {name: c.schema_version for name, c in sorted(CAPABILITIES.items())}
