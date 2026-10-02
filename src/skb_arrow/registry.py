import copy
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from skb_arrow.capabilities import ancombc, echo, mantel
from skb_arrow.errors import HostIncompatible, InvalidParam

# Every JSON type, as messages name it; `float` is a number.
JSON_TYPES = {
    str: "a string",
    int: "an integer",
    float: "a number",
    bool: "a boolean",
    list: "an array",
    dict: "an object",
}
_SCALARS = {str, int, float, bool}
_INT64 = range(-(2**63), 2**63)
_REQUIRED = object()


@dataclass(frozen=True)
class Param:
    kind: type  # a JSON scalar type, or `list`
    default: object = _REQUIRED
    valid: Callable[[Any], bool] | None = None
    rule: str = ""  # what `valid` requires: completes "must be …"
    item: type | None = None  # a `list`'s scalar type

    def __post_init__(self) -> None:
        # So a bad declaration fails at import, not on some call.
        if self.kind not in _SCALARS | {list} or (self.kind is list) != (
            self.item in _SCALARS
        ):
            raise TypeError("a param is a JSON scalar, or an array of one")
        if self.valid is not None and not self.rule:
            raise ValueError("a param's `valid` needs a `rule`")
        if self.default is not _REQUIRED:
            try:
                resolved = _resolve(f"default {self.default!r}", self, self.default)
            except InvalidParam as e:
                raise ValueError(str(e)) from e
            if type(resolved) is not type(self.default) or resolved != self.default:
                raise TypeError(f"default {self.default!r} must be as resolved")


@dataclass(frozen=True)
class Capability:
    schema_version: int
    inputs: frozenset[str]
    params: Mapping[str, Param]
    run: Callable[[Mapping[str, pa.Table], Mapping[str, Any]], pa.Table]


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
    return capability, {
        key: _resolve(f"{name} param {key!r}", param, params.get(key))
        for key, param in capability.params.items()
    }


def schema_versions() -> dict[str, int]:
    return {name: c.schema_version for name, c in sorted(CAPABILITIES.items())}


def _resolve(what: str, param: Param, value: object) -> object:
    """`value` checked against `param`; null, like absent, gives the default."""
    if value is None:
        if param.default is _REQUIRED:
            raise InvalidParam(f"{what} is required")
        return copy.deepcopy(param.default)  # a run can't alter later calls'
    if param.item is None:
        value = _scalar(what, param.kind, value)
    elif type(value) is list:
        value = [
            _scalar(f"{what} item {i}", param.item, v) for i, v in enumerate(value)
        ]
    else:
        raise InvalidParam(f"{what} must be {JSON_TYPES[list]}")
    if param.valid is not None and not param.valid(value):
        raise InvalidParam(f"{what} must be {param.rule}")
    return value


def _scalar(what: str, kind: type, value: object) -> object:
    if kind is float:
        return _number(what, value)
    if type(value) is not kind:
        raise InvalidParam(f"{what} must be {JSON_TYPES[kind]}")
    if kind is int and value not in _INT64:
        raise InvalidParam(f"{what} must be a 64-bit integer")
    return value


def _number(what: str, value: object) -> float:
    """`value` as a finite float: any JSON number, integers included."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidParam(f"{what} must be {JSON_TYPES[float]}")
    try:
        number = float(value)
    except OverflowError:  # an integer beyond float range
        number = math.inf
    if not math.isfinite(number):
        raise InvalidParam(f"{what} must be a finite number")
    return number


# Last: declaring a Param resolves its default.
CAPABILITIES = {
    "ancombc": Capability(
        1,
        frozenset({"table", "metadata"}),
        {
            "formula": Param(str),  # checked against the metadata, in run
            "grouping": Param(str, None),  # likewise
            "posthoc": Param(
                list,
                [],
                lambda v: len(set(v)) == len(v) and set(v) <= set(ancombc.POSTHOC),
                f"distinct, each one of {', '.join(ancombc.POSTHOC)}",
                item=str,
            ),
            "pseudocount": Param(float, 0.0, lambda v: v >= 0, "at least 0"),
            "max_iter": Param(int, 100, lambda v: v >= 1, "at least 1"),
            "tol": Param(float, 1e-5, lambda v: v > 0, "positive"),
            "alpha": Param(float, 0.05, lambda v: 0 < v < 1, "in (0, 1)"),
            "p_adjust": Param(
                str,
                "holm",
                lambda v: v in ancombc.P_ADJUST,
                f"one of {', '.join(ancombc.P_ADJUST)}",
            ),
            "bootstraps": Param(int, 100, lambda v: v >= 1, "at least 1"),
            "seed": Param(int, 0, lambda v: v >= 0, "at least 0"),
        },
        ancombc.run,
    ),
    "echo": Capability(1, frozenset({"table"}), {}, echo.run),
    "mantel": Capability(
        1,
        frozenset({"x", "y"}),
        {
            "method": Param(
                str,
                "pearson",
                lambda v: v in mantel.METHODS,
                f"one of {', '.join(mantel.METHODS)}",
            ),
            "permutations": Param(int, 999, lambda v: v >= 0, "at least 0"),
            "alternative": Param(
                str,
                "two-sided",
                lambda v: v in mantel.ALTERNATIVES,
                f"one of {', '.join(mantel.ALTERNATIVES)}",
            ),
            "seed": Param(int, 0, lambda v: v >= 0, "at least 0"),
        },
        mantel.run,
    ),
}
