import pyarrow as pa
import pytest

from skb_arrow import registry
from skb_arrow.capabilities import echo
from skb_arrow.errors import HostIncompatible, InvalidParam


def test_a_suitable_call_gets_its_capability() -> None:
    assert registry.validate("echo", ["table"], {}) is registry.CAPABILITIES["echo"]


def test_an_unknown_capability_is_host_incompatible() -> None:
    with pytest.raises(HostIncompatible, match="^unknown capability 'nope'$"):
        registry.validate("nope", ["table"], {})


def test_undeclared_params_are_invalid_param() -> None:
    with pytest.raises(InvalidParam, match="^unknown echo params: sed, x$"):
        registry.validate("echo", ["table"], {"x": 1, "sed": 7})


@pytest.mark.parametrize("inputs", [[], ["table", "extra"], ["Table"]])
def test_input_tables_must_be_exactly_those_declared(inputs: list[str]) -> None:
    with pytest.raises(InvalidParam, match="^echo takes inputs: table$"):
        registry.validate("echo", inputs, {})


def test_echo_returns_its_input() -> None:
    table = pa.table({"a": [1]})
    assert echo.run({"table": table}, {}) is table
