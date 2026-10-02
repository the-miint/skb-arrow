"""mantel against direct scikit-bio calls on square, sorted DistanceMatrix inputs."""

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import skbio
import skbio.stats.distance
from client import INIT, SKB_ARROW, Client

from skb_arrow import registry
from skb_arrow.capabilities import mantel
from skb_arrow.errors import classify, collect_warnings

# Descending, so sorting reorders them. Weakly correlated: each alternative's p differs.
IDS = [f"s{i:02d}" for i in range(11, -1, -1)]
_POINTS = np.random.default_rng(3).random((2, len(IDS), 2))
X, Y = (np.hypot(*(p[:, None] - p[None, :]).transpose(2, 0, 1)) for p in _POINTS)
SCHEMA = pa.schema(
    [
        pa.field("statistic", pa.float64(), nullable=False),
        pa.field("pvalue", pa.float64(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
    ]
)
# Every param off its default.
CHANGED: dict[str, Any] = {
    "method": "spearman",
    "permutations": 99,
    "alternative": "greater",
    "seed": 2,
}


def long(square: npt.NDArray[np.float64], ids: list[Any] = IDS) -> pa.Table:
    """`square` as a distance table, each pair once."""
    a, b = np.triu_indices(len(ids), 1)
    return pa.table(
        {
            "id_a": [ids[i] for i in a],
            "id_b": [ids[j] for j in b],
            "distance": square[a, b],
        }
    )


INPUTS = {"x": long(X), "y": long(Y)}


def run(tables: Mapping[str, pa.Table] = INPUTS, **params: object) -> pa.Table:
    capability, resolved = registry.validate("mantel", tables.keys(), params)
    return capability.run(tables, resolved)


def error(tables: Mapping[str, pa.Table] = INPUTS, **params: object) -> tuple[str, str]:
    """The kind and message a call fails with."""
    try:
        run(tables, **params)
    except Exception as e:
        fields = classify(e, capability=True)
        return fields["kind"], fields["message"]
    pytest.fail("the call succeeded")


def expected(
    x: npt.NDArray[np.float64] = X,
    y: npt.NDArray[np.float64] = Y,
    ids: list[Any] = IDS,
    **params: Any,
) -> pa.Table:
    """The natural scikit-bio call: square matrices, IDs sorted."""
    order = sorted(range(len(ids)), key=ids.__getitem__)
    names = [str(ids[i]) for i in order]
    x, y = (skbio.DistanceMatrix(m[np.ix_(order, order)], names) for m in (x, y))
    statistic, pvalue, n = skbio.stats.distance.mantel(
        x, y, engine="numba", **{"seed": 0, **params}
    )
    return pa.table(
        {"statistic": [statistic], "pvalue": [pvalue], "n": [n]}, schema=SCHEMA
    )


@pytest.mark.parametrize(
    ("method", "alternative"),
    list(itertools.product(mantel.METHODS, mantel.ALTERNATIVES)),
)
def test_matches_scikit_bio(method: str, alternative: str) -> None:
    observed = run(method=method, alternative=alternative)
    assert observed.schema == SCHEMA
    assert observed.equals(expected(method=method, alternative=alternative))


@pytest.mark.parametrize("method", mantel.METHODS)
def test_each_alternative_gives_its_own_pvalue_on_this_data(method: str) -> None:
    pvalues = {
        expected(method=method, alternative=a)["pvalue"][0].as_py()
        for a in mantel.ALTERNATIVES
    }
    assert len(pvalues) == len(mantel.ALTERNATIVES)


def test_matches_scikit_bio_with_every_param_changed() -> None:
    assert run(**CHANGED).equals(expected(**CHANGED))


@pytest.mark.parametrize(("param", "value"), CHANGED.items())
def test_each_param_reaches_scikit_bio(param: str, value: Any) -> None:
    reference = expected(**{param: value})
    assert run(**{param: value}).equals(reference)
    assert not reference.equals(expected())  # on this data, alone, it matters


def test_an_omitted_seed_is_seed_0() -> None:
    assert run().equals(run(seed=0))


def test_the_seed_moves_only_the_pvalue() -> None:
    zero, two = run(), run(seed=2)
    assert zero.drop_columns("pvalue").equals(two.drop_columns("pvalue"))
    assert zero["pvalue"] != two["pvalue"]


def test_no_permutations_give_a_nan_pvalue_not_null() -> None:
    pvalue = run(permutations=0)["pvalue"]
    assert pvalue.null_count == 0
    assert np.isnan(pvalue[0].as_py())


@pytest.mark.parametrize(
    ("method", "warned"), [("pearson", True), ("spearman", True), ("kendalltau", False)]
)
def test_a_constant_input_gives_nan_warned_but_by_kendalltau(
    method: str, warned: bool
) -> None:
    tables = {"x": long(np.ones_like(X)), "y": INPUTS["y"]}
    with collect_warnings() as recorded:
        observed = run(tables, method=method)
    assert np.isnan(observed["statistic"][0].as_py())
    assert np.isnan(observed["pvalue"][0].as_py())
    assert observed["statistic"].null_count == observed["pvalue"].null_count == 0
    constant = ["scipy.stats._warnings_errors.ConstantInputWarning"]
    assert [w["category"] for w in recorded] == (constant if warned else [])


def test_row_order_and_orientation_cannot_change_the_answer() -> None:
    rng = np.random.default_rng(1)

    def shuffled(table: pa.Table) -> pa.Table:
        table = table.take(rng.permutation(table.num_rows))
        swap = pa.array(rng.random(table.num_rows) < 0.5)
        a, b = table["id_a"], table["id_b"]
        return table.set_column(0, "id_a", pc.if_else(swap, b, a)).set_column(
            1, "id_b", pc.if_else(swap, a, b)
        )

    tables = {name: shuffled(table) for name, table in INPUTS.items()}
    assert run(tables).equals(run())


def test_integer_ids_sort_as_integers() -> None:
    ids = list(range(100, 88, -1))  # as strings, "100" would sort first
    tables = {"x": long(X, ids), "y": long(Y, ids)}
    assert run(tables).equals(expected(ids=ids))
    assert not expected(ids=ids).equals(expected(ids=[str(i) for i in ids]))


def test_one_thread_gives_every_threads_answer() -> None:
    # scikit-bio's kernel computes each permutation in one thread, so this holds today;
    # it pins the documented promise across scikit-bio upgrades.
    threads = numba.get_num_threads()
    assert threads > 1, "needs more than one core to mean anything"
    every = run()
    try:
        numba.set_num_threads(1)
        assert run().equals(every)
    finally:
        numba.set_num_threads(threads)


def test_numba_runs_the_permutations(monkeypatch: pytest.MonkeyPatch) -> None:
    module = skbio.stats.distance._mantel
    kernel = module._mantel_perm_pearsonr_condensed_nb
    calls: list[object] = []

    def spy(*args: object) -> object:
        calls.append(args)
        return kernel(*args)

    monkeypatch.setattr(module, "_mantel_perm_pearsonr_condensed_nb", spy)
    run()
    assert len(calls) == 1
    assert kernel.signatures  # compiled, not run as Python


@pytest.mark.parametrize(
    ("param", "value", "rule"),
    [
        ("method", "kendall", "one of pearson, spearman, kendalltau"),
        ("permutations", -1, "at least 0"),
        ("alternative", "two_sided", "one of two-sided, greater, less"),
        ("seed", -1, "at least 0"),
    ],
)
def test_a_param_breaking_its_rule_is_invalid_param(
    param: str, value: object, rule: str
) -> None:
    assert error(INPUTS, **{param: value}) == (
        "invalid_param",
        f"mantel param {param!r} must be {rule}",
    )


def test_each_rules_bound_is_accepted() -> None:
    registry.validate("mantel", INPUTS, {"permutations": 0, "seed": 0})


def test_inputs_over_different_ids_are_invalid_input_naming_them() -> None:
    ids = [*IDS[:-1], "t"]
    tables = {"x": INPUTS["x"], "y": long(Y, ids)}
    assert error(tables) == ("invalid_input", "y: 1 of 12 IDs are not in x, e.g. 't'")


def test_x_then_y_then_their_ids_are_checked() -> None:
    empty, negative = INPUTS["x"].slice(0, 0), long(-Y, [*IDS[1:], "t"])
    assert error({"x": empty, "y": negative}) == ("invalid_input", "x: no rows")
    assert error({"x": INPUTS["x"], "y": negative}) == (
        "invalid_input",
        "y: 66 of 66 distances are negative or not finite, e.g. ('s00', 's01'), "
        "('s00', 's02'), ('s00', 's03'), ('s00', 's04'), ('s00', 's05')",
    )


def test_fewer_than_three_ids_is_scikit_bios_invalid_input() -> None:
    pair = long(X[:2, :2], IDS[:2])
    assert error({"x": pair, "y": pair}) == (
        "invalid_input",
        "ValueError: Distance matrices must have at least 3 matching IDs between them "
        "(i.e., minimum 3x3 in size).",
    )


def test_end_to_end_across_segments(tmp_path: Path) -> None:
    # numba compiles in the host: both CI platforms run it.
    directory = tmp_path / "session"
    directory.mkdir()
    client = Client(SKB_ARROW, directory)
    try:
        assert client.send(INIT | {"segment_bytes": 1024})["type"] == "ready"
        inputs = {}
        for name, table in INPUTS.items():
            rows = -(-table.num_rows // 3)
            inputs[name] = [
                client.put(f"{name}-{i}", table.slice(i * rows, rows)) for i in range(3)
            ]
        response = client.send(
            {
                "type": "call",
                "capability": "mantel",
                "params": CHANGED,
                "input": inputs,
            }
        )
        assert response["type"] == "result", response
        assert client.fetch(response["output"]).equals(expected(**CHANGED))
        assert client.shut() == 0
    finally:
        client.kill()
