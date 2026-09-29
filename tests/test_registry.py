import json
import sys
from collections.abc import Callable

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
MORE = {
    "flag": registry.Param(bool, False),
    "seed": registry.Param(int, 0),
    "tags": registry.Param(
        list, [], lambda v: len(set(v)) == len(v), "distinct", item=str
    ),
}


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, params in [("probe", PARAMS), ("more", MORE)]:
        capability = registry.Capability(1, frozenset({"table"}), params, echo.run)
        monkeypatch.setitem(registry.CAPABILITIES, name, capability)


def resolve(params: dict[str, object], name: str = "probe") -> dict[str, object]:
    return registry.validate(name, ["table"], params)[1]


def test_a_suitable_call_gets_its_capability() -> None:
    capability, params = registry.validate("echo", ["table"], {})
    assert (capability, params) == (registry.CAPABILITIES["echo"], {})
    assert capability is registry.CAPABILITIES["echo"]


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


def test_param_names_are_checked_before_inputs() -> None:
    with pytest.raises(InvalidParam, match="^unknown probe params: sed$"):
        registry.validate("probe", [], {"sed": 1})


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


def test_a_mistyped_boolean_is_invalid() -> None:
    with pytest.raises(InvalidParam, match="^more param 'flag' must be a boolean$"):
        resolve({"flag": 1}, "more")


@pytest.mark.parametrize("seed", [-(2**63), 2**63 - 1])
def test_an_integer_may_span_64_bits(seed: int) -> None:
    assert resolve({"seed": seed}, "more")["seed"] == seed


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**63])
def test_an_integer_must_fit_64_bits(seed: int) -> None:
    with pytest.raises(InvalidParam, match="^more param 'seed' must be a 64-bit"):
        resolve({"seed": seed}, "more")


def test_an_array_holds_items_of_its_type() -> None:
    assert resolve({"tags": ["a", "b"]}, "more")["tags"] == ["a", "b"]
    with pytest.raises(InvalidParam, match="^more param 'tags' must be an array$"):
        resolve({"tags": "a"}, "more")
    for tags, index in [(["a", 1], 1), ([{}], 0), ([None], 0)]:
        item = f"^more param 'tags' item {index} must be a string$"
        with pytest.raises(InvalidParam, match=item):
            resolve({"tags": tags}, "more")


def test_an_array_rule_sees_typed_items() -> None:
    with pytest.raises(InvalidParam, match="^more param 'tags' must be distinct$"):
        resolve({"tags": ["a", "a"]}, "more")


def test_each_call_gets_its_own_default() -> None:
    tags = resolve({}, "more")["tags"]
    assert isinstance(tags, list)
    tags.append("x")  # as a run might
    assert resolve({}, "more")["tags"] == []


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


def test_an_integer_that_rounds_to_a_finite_float_is_a_number() -> None:
    rate = resolve({"name": "a", "rate": int(sys.float_info.max) + 1})["rate"]
    assert rate == sys.float_info.max


def test_a_param_must_pass_its_rule() -> None:
    assert resolve({"name": "a", "count": 1})["count"] == 1
    with pytest.raises(InvalidParam, match="^probe param 'count' must be at least 1$"):
        resolve({"name": "a", "count": 0})


@pytest.mark.parametrize(
    "declare",
    [
        lambda: registry.Param(dict),
        lambda: registry.Param(list),
        lambda: registry.Param(list, item=list),
        lambda: registry.Param(str, item=str),
        lambda: registry.Param(float, 0),
    ],
    ids=["dict", "array without item", "array of arrays", "item on scalar", "0 as 0.0"],
)
def test_a_badly_typed_declaration_fails_at_once(
    declare: Callable[[], registry.Param],
) -> None:
    with pytest.raises(TypeError):
        declare()


@pytest.mark.parametrize(
    ("declare", "message"),
    [
        (lambda: registry.Param(int, 3, lambda n: n >= 1), "`valid` needs a `rule`"),
        (lambda: registry.Param(int, 0, lambda n: n >= 1, "positive"), "default 0 "),
        (lambda: registry.Param(list, [1], item=str), "default \\[1\\] item 0"),
    ],
    ids=["no rule", "default breaks rule", "default mistyped"],
)
def test_a_bad_declaration_fails_at_once(
    declare: Callable[[], registry.Param], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        declare()


def test_echo_returns_its_input() -> None:
    table = pa.table({"a": [1]})
    assert echo.run({"table": table}, {}) is table
