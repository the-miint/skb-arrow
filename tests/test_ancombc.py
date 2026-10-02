"""ancombc against direct scikit-bio calls, on the HITChip Atlas data (data/atlas)."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import skbio.stats.composition
from client import INIT, SKB_ARROW, Client

from skb_arrow import registry
from skb_arrow.errors import classify

DATA = Path(__file__).with_name("data") / "atlas"
COUNTS = pd.read_csv(DATA / "atlas_feature_table.csv", index_col=0)
META = pd.read_csv(DATA / "atlas_meta_data.csv", index_col=0)
BMI = ["lean", "overweight", "obese"]
STRING = pa.string()
FORMULA = "age + region + bmi"
EVERY_TEST = ["global", "pairwise", "dunnett"]
# Every param off its default. At this alpha the global screen keeps features.
CHANGED: dict[str, Any] = {
    "max_iter": 2,
    "tol": 1e-2,
    "alpha": 0.5,
    "p_adjust": "bh",
    "bootstraps": 10,
    "seed": 2,
}
PWNED = "SKB_ARROW_PWNED"  # set only if a formula's code runs
PAYLOAD = f"__import__('os').environ.setdefault({PWNED!r}, '1')"


def long(counts: pd.DataFrame = COUNTS, feature_id: pa.DataType = STRING) -> pa.Table:
    """`counts` as a feature table, its zero cells absent."""
    cells = counts.stack()
    cells = cells[cells != 0]
    return pa.table(
        {
            "sample_id": cells.index.get_level_values(0).tolist(),
            "feature_id": pa.array(
                cells.index.get_level_values(1).tolist(), feature_id
            ),
            "value": cells.tolist(),
        }
    )


def metadata(meta: pd.DataFrame = META) -> pa.Table:
    """`meta` with `bmi` a dictionary, lean first: the reference level."""
    columns: dict[str, Any] = {"sample_id": meta.index.tolist()}
    for name in meta.columns:
        values = meta[name]
        columns[name] = (
            pa.array(pd.Categorical(values, categories=BMI))
            if name == "bmi"
            else values.tolist()
        )
    return pa.table(columns)


INPUTS = {"table": long(), "metadata": metadata()}


def ancombc(tables: Mapping[str, pa.Table] = INPUTS, **params: object) -> pa.Table:
    params = {"formula": FORMULA, **params}
    capability, resolved = registry.validate("ancombc", tables.keys(), params)
    return capability.run(tables, resolved)


def error(tables: Mapping[str, pa.Table] = INPUTS, **params: object) -> tuple[str, str]:
    """The kind and message a call fails with."""
    try:
        ancombc(tables, **params)
    except Exception as e:
        fields = classify(e, capability=True)
        return fields["kind"], fields["message"]
    pytest.fail("the call succeeded")


def schema(feature_id: pa.DataType) -> pa.Schema:
    return pa.schema(
        [
            pa.field("feature_id", feature_id, nullable=False),
            pa.field("test", pa.string(), nullable=False),
            pa.field("term", pa.string()),
            pa.field("lfc", pa.float64()),
            pa.field("se", pa.float64()),
            pa.field("w", pa.float64(), nullable=False),
            pa.field("pvalue", pa.float64(), nullable=False),
            pa.field("qvalue", pa.float64(), nullable=False),
            pa.field("signif", pa.bool_(), nullable=False),
        ]
    )


def expected(
    counts: pd.DataFrame = COUNTS,
    feature_id: pa.DataType = STRING,
    *,
    posthoc: list[str] = EVERY_TEST,
    grouping: str | None = "bmi",
    bootstraps: int = 100,
    seed: int = 0,
    **params: Any,
) -> pa.Table:
    """The natural scikit-bio call on DataFrames, its frames as one output table."""
    counts = counts.sort_index().sort_index(axis=1) + 1
    meta = META.loc[counts.index].copy()
    meta["bmi"] = pd.Categorical(meta["bmi"], categories=BMI)
    result = skbio.stats.composition.ancombc(
        counts, meta, FORMULA, grouping=grouping, **params
    )
    tests = {
        "global": result.global_test,
        "pairwise": result.pairwise_test,
        "dunnett": lambda: result.dunnett_test(bootstraps=bootstraps, seed=seed),
    }
    frames = [("main", result.result)]
    frames += [(test, run()) for test, run in tests.items() if test in posthoc]
    rows: dict[str, list[Any]] = {name: [] for name in schema(feature_id).names}
    for test, frame in frames:
        n = len(frame)
        termed = frame.index.nlevels == 2
        rows["feature_id"] += frame.index.get_level_values(0).tolist()
        rows["test"] += [test] * n
        rows["term"] += (
            frame.index.get_level_values(1).tolist() if termed else [None] * n
        )
        rows["lfc"] += frame["Log(FC)"].tolist() if termed else [None] * n
        rows["se"] += frame["SE"].tolist() if termed else [None] * n
        for name, column in [("w", "W"), ("pvalue", "pvalue"), ("qvalue", "qvalue")]:
            rows[name] += frame[column].tolist()
        rows["signif"] += frame["Signif"].tolist()
    return pa.table(rows, schema=schema(feature_id))


def test_matches_scikit_bio_with_every_posthoc_test() -> None:
    observed = ancombc(pseudocount=1, grouping="bmi", posthoc=EVERY_TEST)
    assert observed.schema == schema(pa.string())
    assert observed.equals(expected())


def test_matches_scikit_bio_with_every_param_changed() -> None:
    observed = ancombc(pseudocount=1, grouping="bmi", posthoc=EVERY_TEST, **CHANGED)
    reference = expected(**CHANGED)
    assert observed.equals(reference)
    # Not the degenerate case: at the default alpha every pairwise p-value is 1.
    pairwise = reference.filter(pc.equal(reference["test"], "pairwise"))
    assert len(pc.unique(pairwise["pvalue"])) > 1


@pytest.mark.parametrize(("param", "value"), CHANGED.items())
def test_each_param_reaches_scikit_bio(param: str, value: Any) -> None:
    observed = ancombc(
        pseudocount=1, grouping="bmi", posthoc=EVERY_TEST, **{param: value}
    )
    reference = expected(**{param: value})
    assert observed.equals(reference)
    assert not reference.equals(expected())  # on this data, alone, it matters


def test_by_default_only_the_main_test_runs() -> None:
    assert ancombc(pseudocount=1).equals(expected(posthoc=[], grouping=None))


@pytest.mark.parametrize("posthoc", [["dunnett"], ["dunnett", "global"]])
def test_runs_only_the_posthoc_tests_asked_for_in_a_fixed_order(
    posthoc: list[str],
) -> None:
    observed = ancombc(pseudocount=1, grouping="bmi", posthoc=posthoc)
    assert observed.equals(expected(posthoc=posthoc))


def test_integer_feature_ids_come_back_as_int64() -> None:
    # Descending, so sorting by ID reorders the features.
    counts = COUNTS.set_axis(range(100, 100 - COUNTS.shape[1], -1), axis=1)
    tables = {"table": long(counts, pa.int32()), "metadata": metadata()}
    observed = ancombc(tables, pseudocount=1, grouping="bmi", posthoc=EVERY_TEST)
    assert observed.schema == schema(pa.int64())
    assert observed.equals(expected(counts, pa.int64()))


def test_agrees_with_r_ancombc() -> None:
    observed = ancombc(pseudocount=1, grouping="bmi", posthoc=["global"]).to_pandas()
    for test, key in [("main", ["feature_id", "term"]), ("global", ["feature_id"])]:
        r = pd.read_table(DATA / f"atlas_ancombc_{test}.tsv")
        r.columns = [*key, *r.columns[len(key) :]]
        ours = observed[observed["test"] == test].set_index(key)
        r = r.set_index(key)
        assert sorted(ours.index) == sorted(r.index)
        ours = ours.loc[r.index]
        pairs = [("W", "w"), ("pvalue", "pvalue"), ("qvalue", "qvalue")]
        if test == "main":
            pairs += [("Log(FC)", "lfc"), ("SE", "se")]
        for theirs, name in pairs:
            np.testing.assert_allclose(ours[name], r[theirs], rtol=0, atol=1e-3)
        assert (ours["signif"] == r["Signif"]).all()


def test_a_repeated_request_gives_an_identical_table() -> None:
    params: dict[str, Any] = {
        "pseudocount": 1,
        "grouping": "bmi",
        "posthoc": EVERY_TEST,
    }
    assert ancombc(**params).equals(ancombc(**params))


def test_row_order_cannot_change_the_answer() -> None:
    rng = np.random.default_rng(1)
    shuffled = {
        name: table.take(rng.permutation(table.num_rows))
        for name, table in INPUTS.items()
    }
    params: dict[str, Any] = {
        "pseudocount": 1,
        "grouping": "bmi",
        "posthoc": EVERY_TEST,
    }
    assert ancombc(shuffled, **params).equals(ancombc(**params))


def test_an_omitted_seed_is_seed_0() -> None:
    params: dict[str, Any] = {
        "pseudocount": 1,
        "grouping": "bmi",
        "posthoc": ["dunnett"],
    }
    assert ancombc(**params).equals(ancombc(**params, seed=0))


def test_the_seed_changes_only_the_dunnett_rows() -> None:
    params: dict[str, Any] = {
        "pseudocount": 1,
        "grouping": "bmi",
        "posthoc": EVERY_TEST,
    }
    zero, two = ancombc(**params), ancombc(**params, seed=2)
    dunnett = pc.equal(zero["test"], "dunnett")
    assert zero.filter(pc.invert(dunnett)).equals(two.filter(pc.invert(dunnett)))
    assert not zero.filter(dunnett).equals(two.filter(dunnett))


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("age +", "does not parse: "),
        # patsy's tokenizer asserts, and its parser recurses per term.
        pytest.param("age\0", "does not parse", id="NUL"),
        pytest.param("(" * 250 + "age" + ")" * 250, "does not parse", id="nested"),
        pytest.param(" + ".join(["age"] * 2000), "does not parse", id="2000 terms"),
        ("", "names no metadata column"),
        ("   ", "names no metadata column"),
        ("0", "names no metadata column"),
        ("1", "names no metadata column"),
        ("bmi ~ age", "must have no left-hand side"),
        ("age + log(age)", "names 'log(age)', not a bare column name"),
        ("age + table", "names 'table', not a metadata column"),
    ],
)
def test_a_bad_formula_is_invalid_param(formula: str, message: str) -> None:
    kind, text = error(pseudocount=1, formula=formula)
    assert kind == "invalid_param"
    assert text.startswith("ancombc param 'formula' ")
    assert message in text


@pytest.mark.parametrize("name", ["ｔａｂｌｅ", "__debug__", "None", PAYLOAD])
def test_a_column_python_cannot_name_barely_is_not_a_factor(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PWNED, raising=False)
    tables = {"table": long(), "metadata": metadata(META.assign(**{name: META["age"]}))}
    assert error(tables, pseudocount=1, formula=f"age + {name}") == (
        "invalid_param",
        f"ancombc param 'formula' names {name!r}, not a bare column name",
    )
    assert PWNED not in os.environ


@pytest.mark.parametrize(
    "params",
    [{"formula": "age +"}, {"posthoc": ["global"]}],
    ids=["formula", "posthoc"],
)
def test_what_needs_no_data_is_checked_before_the_tables(
    params: dict[str, Any],
) -> None:
    # Without a pseudocount the table is invalid too.
    assert error(**params)[0] == "invalid_param"


def test_a_left_hand_side_is_never_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PWNED, raising=False)
    assert error(pseudocount=1, formula=f"{PAYLOAD} ~ age") == (
        "invalid_param",
        "ancombc param 'formula' must have no left-hand side",
    )
    assert PWNED not in os.environ


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"grouping": "nope"},
            "ancombc param 'grouping' names 'nope', not a metadata column",
        ),
        (
            {"grouping": "age"},
            "ancombc param 'grouping' must be strings or a dictionary",
        ),
        (
            {"grouping": "bmi", "formula": "age + region + bmi:region"},
            "ancombc param 'grouping' must be a term of its own in 'formula'",
        ),
        (
            {"grouping": "bmi", "formula": "0 + age + bmi"},
            "ancombc param 'grouping' needs an intercept in 'formula'",
        ),
        ({"posthoc": ["global"]}, "ancombc param 'posthoc' needs 'grouping'"),
    ],
    ids=["not a column", "numeric", "interaction only", "no intercept", "no grouping"],
)
def test_a_bad_grouping_is_invalid_param(params: dict[str, Any], message: str) -> None:
    assert error(pseudocount=1, **params) == ("invalid_param", message)


@pytest.mark.parametrize(
    ("param", "value", "rule"),
    [
        (
            "posthoc",
            ["global", "global"],
            "distinct, each one of global, pairwise, dunnett",
        ),
        ("posthoc", ["trend"], "distinct, each one of global, pairwise, dunnett"),
        ("pseudocount", -0.5, "at least 0"),
        ("max_iter", 0, "at least 1"),
        ("tol", 0, "positive"),
        ("alpha", 0, "in (0, 1)"),
        ("alpha", 1, "in (0, 1)"),
        ("p_adjust", "fdr_bh", "one of holm, bonferroni, bh, by"),
        ("bootstraps", 0, "at least 1"),
        ("seed", -1, "at least 0"),
    ],
)
def test_a_param_breaking_its_rule_is_invalid_param(
    param: str, value: object, rule: str
) -> None:
    assert error(INPUTS, **{param: value}) == (
        "invalid_param",
        f"ancombc param {param!r} must be {rule}",
    )


def test_each_rules_bound_is_accepted() -> None:
    params = {
        "pseudocount": 0,
        "max_iter": 1,
        "tol": 5e-324,
        "bootstraps": 1,
        "seed": 0,
    }
    for alpha in [5e-324, 1 - 2**-53]:
        registry.validate("ancombc", INPUTS, {"formula": "a", "alpha": alpha, **params})


def test_a_boolean_grouping_is_invalid_param() -> None:
    # It has two groups at most; scikit-bio needs three.
    tables = {"table": long(), "metadata": metadata(META.assign(old=META["age"] > 40))}
    assert error(tables, pseudocount=1, formula="age + old", grouping="old") == (
        "invalid_param",
        "ancombc param 'grouping' must be strings or a dictionary",
    )


def test_fewer_than_three_groups_is_scikit_bios_invalid_input() -> None:
    meta = META.assign(bmi=META["bmi"].replace("obese", "overweight"))
    tables = {"table": long(), "metadata": metadata(meta)}
    assert error(tables, pseudocount=1, grouping="bmi") == (
        "invalid_input",
        "ValueError: `grouping` must contain at least three observed groups.",
    )


def test_a_zero_cell_without_a_pseudocount_is_invalid_input_naming_it() -> None:
    counts = COUNTS.sort_index().sort_index(axis=1)
    zeros = counts.stack()
    zeros = zeros[zeros == 0]
    kind, message = error()
    assert kind == "invalid_input"
    assert message.startswith(
        f"table: {len(zeros)} of {counts.size} cells are not positive and finite, "
        f"e.g. {zeros.index[0]!r}, "
    )


def test_a_sample_missing_from_the_metadata_is_invalid_input_naming_it() -> None:
    tables = {"table": long(), "metadata": metadata(META.drop(index="Sample-8"))}
    assert error(tables, pseudocount=1) == (
        "invalid_input",
        "metadata: 1 of 300 table samples are missing, e.g. 'Sample-8'",
    )


def test_end_to_end_across_segments(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    client = Client(SKB_ARROW, directory)
    try:
        assert client.send(INIT | {"segment_bytes": 1024})["type"] == "ready"
        params = {
            "formula": FORMULA,
            "grouping": "bmi",
            "posthoc": EVERY_TEST,
            "pseudocount": 1,
        }
        response = client.call("ancombc", INPUTS, params, parts=3)
        assert response["type"] == "result", response
        assert len(response["output"]) > 1
        assert client.fetch(response["output"]).equals(expected())
        assert client.shut() == 0
    finally:
        client.kill()
