from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

import pyarrow as pa

from skb_arrow.capabilities import echo
from skb_arrow.errors import HostIncompatible, InvalidParam


@dataclass(frozen=True)
class Capability:
    schema_version: int
    inputs: frozenset[str]
    params: frozenset[str]
    run: Callable[[Mapping[str, pa.Table], Mapping[str, object]], pa.Table]


CAPABILITIES = {"echo": Capability(1, frozenset({"table"}), frozenset(), echo.run)}


def validate(
    name: str, inputs: Collection[str], params: Mapping[str, object]
) -> Capability:
    """Capability `name`, once a call's input table names and params suit it."""
    capability = CAPABILITIES.get(name)
    if capability is None:
        raise HostIncompatible(f"unknown capability {name!r}")
    if unknown := params.keys() - capability.params:
        raise InvalidParam(f"unknown {name} params: {', '.join(sorted(unknown))}")
    if set(inputs) != capability.inputs:
        raise InvalidParam(
            f"{name} takes inputs: {', '.join(sorted(capability.inputs))}"
        )
    return capability


def schema_versions() -> dict[str, int]:
    return {name: c.schema_version for name, c in sorted(CAPABILITIES.items())}
