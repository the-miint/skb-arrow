import json

import pyarrow as pa
import pytest

from skb_arrow import registry
from skb_arrow.capabilities import echo
from skb_arrow.errors import HostIncompatible, InvalidParam

PARAMS = {
    "count": registry.Param(int, 3, lambda n: n >= 1, "at least 1"),
    "rate": registry.Param(float, 0.5),
    "name": registry.Param(str),
}


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = registry.Capability(1, frozenset({"table"}), PARAMS, echo.run)
    monkeypatch.setitem(registry.CAPABILITIES, "probe", capability)


def resolve(params: dict[str, object]) -> dict[str, object]:
    return registry.validate("probe", ["table"], params)[1]


def test_a_suitable_call_gets_its_capability() -> None:
    assert registry.validate("echo", ["table"], {}) == (
        registry.CAPABILITIES["echo"],
        {},
    )


def test_an_unknown_capability_is_host_incompatible() -> None:
    with pytest.raises(HostIncompatible, match="^unknown capability 'nope'$"):
        registry.validate("nope", ["table"], {})


def test_undeclared_params_are_invalid_param() -> None:
    with pytest.raises(InvalidParam, match="^unknown echo params: sed, x$"):
        registry.validate("echo", ["table"], {"x": 1, "sed": 7})


def test_an_undeclared_param_is_invalid_even_when_null() -> None:
    with pytest.raises(InvalidParam, match="^unknown probe params: sed$"):
        resolve({"name": "a", "sed": None})


@pytest.mark.parametrize("inputs", [[], ["table", "extra"], ["Table"]])
def test_input_tables_must_be_exactly_those_declared(inputs: list[str]) -> None:
    with pytest.raises(InvalidParam, match="^echo takes inputs: table$"):
        registry.validate("echo", inputs, {})


def test_names_are_checked_before_values() -> None:
    with pytest.raises(InvalidParam, match="^unknown probe params: sed$"):
        resolve({"name": 1, "sed": 1})
    with pytest.raises(InvalidParam, match="^probe takes inputs: table$"):
        registry.validate("probe", [], {})


def test_every_param_is_resolved_with_defaults_filled() -> None:
    assert resolve({"name": "a"}) == {"count": 3, "rate": 0.5, "name": "a"}
    assert resolve({"name": "a", "count": 5}) == {"count": 5, "rate": 0.5, "name": "a"}


def test_a_null_param_takes_its_default() -> None:
    assert resolve({"name": "a", "count": None})["count"] == 3


@pytest.mark.parametrize("params", [{}, {"name": None}], ids=["absent", "null"])
def test_a_param_without_default_is_required(params: dict[str, object]) -> None:
    with pytest.raises(InvalidParam, match="^probe param 'name' is required$"):
        resolve(params)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("count", True, "an integer"),
        ("count", 1.5, "an integer"),
        ("count", "3", "an integer"),
        ("rate", True, "a number"),
        ("rate", "0.5", "a number"),
        ("name", 1, "a string"),
    ],
    ids=repr,
)
def test_a_mistyped_param_is_invalid(key: str, value: object, expected: str) -> None:
    with pytest.raises(InvalidParam, match=f"^probe param '{key}' must be {expected}$"):
        resolve({"name": "a", key: value})


def test_a_number_takes_an_integer_as_a_float() -> None:
    rate = resolve({"name": "a", "rate": 2})["rate"]
    assert (type(rate), rate) == (float, 2.0)


@pytest.mark.parametrize(
    "rate",
    [json.loads("1e400"), json.loads("-1e400"), 10**400, -(10**400)],
    ids=["1e400", "-1e400", "10**400", "-10**400"],
)
def test_a_number_must_be_finite(rate: float) -> None:
    finite = "^probe param 'rate' must be a finite number$"
    with pytest.raises(InvalidParam, match=finite):
        resolve({"name": "a", "rate": rate})


def test_a_param_must_pass_its_rule() -> None:
    assert resolve({"name": "a", "count": 1})["count"] == 1
    with pytest.raises(InvalidParam, match="^probe param 'count' must be at least 1$"):
        resolve({"name": "a", "count": 0})


def test_echo_returns_its_input() -> None:
    table = pa.table({"a": [1]})
    assert echo.run({"table": table}, {}) is table
