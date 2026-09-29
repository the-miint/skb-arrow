from collections.abc import Callable
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from skb_arrow.capabilities import _tables
from skb_arrow.errors import InvalidInput

Rows = list[tuple[object, object, object]]
DENSE: Rows = [("s2", "f2", 4), ("s1", "f2", 2), ("s2", "f1", 3), ("s1", "f1", 1)]
MATRIX = [[1.0, 2.0], [3.0, 4.0]]


STRING, INT64 = pa.string(), pa.int64()


def long(
    rows: Rows,
    sample: pa.DataType = STRING,
    feature: pa.DataType = STRING,
    value: pa.DataType = INT64,
) -> pa.Table:
    samples, features, values = zip(*rows, strict=True) if rows else ((), (), ())
    return pa.table(
        {
            "sample_id": pa.array(samples, sample),
            "feature_id": pa.array(features, feature),
            "value": pa.array(values, value),
        }
    )


def load(table: pa.Table, pseudocount: float = 0.0) -> _tables.FeatureTable:
    return _tables.feature_table({"table": table}, "table", pseudocount=pseudocount)


def rejects(message: str) -> pytest.RaisesExc[InvalidInput]:
    return pytest.raises(InvalidInput, match=f"^{message}$")


def test_a_table_becomes_its_matrix_with_ids_sorted() -> None:
    loaded = load(long(DENSE))
    assert loaded.samples.equals(pa.array(["s1", "s2"]))
    assert loaded.features.equals(pa.array(["f1", "f2"]))
    assert np.array_equal(loaded.matrix, MATRIX)
    assert loaded.matrix.dtype == np.float64


def test_the_matrix_is_in_fortran_order() -> None:
    assert load(long(DENSE)).matrix.flags.f_contiguous


def test_row_order_never_changes_the_result() -> None:
    first, second = load(long(DENSE)), load(long(DENSE[::-1]))
    assert np.array_equal(first.matrix, second.matrix)
    assert first.samples.equals(second.samples)
    assert first.features.equals(second.features)


def test_pseudocount_fills_absent_cells_and_adds_to_present_ones() -> None:
    loaded = load(long(DENSE[1:]), pseudocount=0.5)
    assert np.array_equal(loaded.matrix, [[1.5, 2.5], [3.5, 0.5]])


@pytest.mark.parametrize(
    "columns",
    [
        ["sample_id", "feature_id"],
        ["sample_id", "feature_id", "value", "taxon"],
        ["sample_id", "feature_id", "count"],
        ["sample_id", "feature_id", "value", "value"],
    ],
    ids=["missing", "extra", "renamed", "repeated"],
)
def test_the_columns_are_exactly_the_three(columns: list[str]) -> None:
    table = pa.table([pa.array(["s1"])] * len(columns), names=columns)
    got = ", ".join(columns)
    with rejects(f"table: columns must be sample_id, feature_id, value; got {got}"):
        load(table)


def test_the_columns_may_come_in_any_order() -> None:
    table = long(DENSE).select(["value", "feature_id", "sample_id"])
    assert np.array_equal(load(table).matrix, MATRIX)


@pytest.mark.parametrize(
    ("column", "ids"),
    [("sample_id", pa.array([1.0, 2.0] * 2)), ("feature_id", pa.array([True] * 4))],
    ids=["float samples", "bool features"],
)
def test_ids_are_integers_or_strings(column: str, ids: pa.Array) -> None:
    table = long(DENSE)
    table = table.set_column(table.column_names.index(column), column, ids)
    with rejects(f"table: {column} must be integers or strings, not {ids.type}"):
        load(table)


INTEGER: Rows = [(2, 20, 4), (1, 20, 2), (2, 10, 3), (1, 10, 1)]
NORMALIZED = {
    "large_string": (DENSE, lambda a: a.cast(pa.large_string())),
    "string_view": (DENSE, lambda a: a.cast(pa.string_view())),
    "dictionary": (DENSE, lambda a: a.dictionary_encode()),
    "int32": (INTEGER, lambda a: a.cast(pa.int32())),
    "uint8": (INTEGER, lambda a: a.cast(pa.uint8())),
    "int16 dictionary": (INTEGER, lambda a: a.cast(pa.int16()).dictionary_encode()),
}


@pytest.mark.parametrize(("rows", "convert"), NORMALIZED.values(), ids=NORMALIZED)
def test_ids_are_normalized(
    rows: Rows, convert: Callable[[pa.Array], pa.Array]
) -> None:
    kind = pa.int64() if rows is INTEGER else pa.string()
    table = long(rows, kind, kind)
    for i in range(2):
        column = convert(table.column(i).chunk(0))
        table = table.set_column(i, table.field(i).name, column)
    loaded, plain = load(table), load(long(rows, kind, kind))
    assert loaded.samples.equals(plain.samples)
    assert loaded.features.equals(plain.features)
    assert (loaded.samples.type, loaded.features.type) == (kind, kind)
    assert np.array_equal(loaded.matrix, MATRIX)


def test_an_integer_id_must_fit_int64() -> None:
    with rejects("table: sample_id must fit int64"):
        load(long([(2**64 - 1, "f1", 1)], sample=pa.uint64()))


@pytest.mark.parametrize(
    "values",
    [
        pa.array([Decimal(v) for v in range(4)], pa.decimal128(5, 0)),
        pa.array([True] * 4),
        pa.array(["1"] * 4),
    ],
    ids=["decimal", "bool", "string"],
)
def test_values_are_integers_or_floating_point(values: pa.Array) -> None:
    table = long(DENSE).set_column(2, "value", values)
    kind = str(values.type).replace("(", r"\(").replace(")", r"\)")
    with rejects(f"table: value must be integers or floating point, not {kind}"):
        load(table)


def test_floating_point_values_keep_their_fractions() -> None:
    rows: Rows = [("s2", "f2", 4.25), ("s1", "f2", 2.25), ("s2", "f1", 3.25)]
    loaded = load(long([*rows, ("s1", "f1", 1.25)], value=pa.float32()))
    assert np.array_equal(loaded.matrix, [[1.25, 2.25], [3.25, 4.25]])


@pytest.mark.parametrize("column", ["sample_id", "feature_id"])
def test_an_id_is_never_null(column: str) -> None:
    extra = (None, "f1", 1) if column == "sample_id" else ("s1", None, 1)
    with rejects(f"table: {column} is null in 1 of 5 rows"):
        load(long([*DENSE, extra]))


def test_a_value_is_never_null() -> None:
    with rejects(r"table: value is null in 1 of 4 rows, e.g. \('s1', 'f2'\)"):
        load(long([DENSE[0], ("s1", "f2", None), *DENSE[2:]]))


def test_null_values_are_shown_in_sorted_order() -> None:
    rows: Rows = [("s2", "f2", None), ("s1", "f2", None), *DENSE[2:]]
    shown = r"\('s1', 'f2'\), \('s2', 'f2'\)"
    with rejects(f"table: value is null in 2 of 4 rows, e.g. {shown}"):
        load(long(rows))


def test_a_table_has_rows() -> None:
    with rejects("table: no rows"):
        load(long([]))


def test_a_pair_appears_once() -> None:
    rows = [*DENSE, ("s2", "f1", 7), ("s1", "f2", 5), ("s1", "f2", 6)]
    shown = r"\('s1', 'f2'\), \('s2', 'f1'\)"
    with rejects(
        rf"table: 2 of 4 \(sample_id, feature_id\) pairs repeat, e.g. {shown}"
    ):
        load(long(rows))


NOT_POSITIVE = r"cells are not positive and finite"


@pytest.mark.parametrize(
    "value",
    [None, 0.0, -1.0, float("nan"), float("inf")],
    ids=["absent", "zero", "negative", "nan", "inf"],
)
def test_every_cell_is_finite_and_positive(value: float | None) -> None:
    rows: Rows = [("s2", "f2", value), *DENSE[1:]] if value is not None else DENSE[1:]
    with rejects(rf"table: 1 of 4 {NOT_POSITIVE}, e.g. \('s2', 'f2'\)"):
        load(long(rows, value=pa.float64()))


def test_a_pseudocount_does_not_excuse_a_negative_cell() -> None:
    with rejects(rf"table: 1 of 4 {NOT_POSITIVE}, e.g. \('s2', 'f2'\)"):
        load(long([("s2", "f2", -1), *DENSE[1:]]), pseudocount=0.5)


def test_a_message_counts_every_offence_but_shows_five() -> None:
    rows: Rows = [("s1", f"f{i}", 0 if i < 7 else 1) for i in range(8)]
    shown = ", ".join(rf"\('s1', 'f{i}'\)" for i in range(5))
    with rejects(f"table: 7 of 8 {NOT_POSITIVE}, e.g. {shown}"):
        load(long(rows))


def test_examples_come_in_sorted_order() -> None:
    rows: Rows = [("s2", "f1", 0), ("s1", "f2", 0), ("s1", "f1", 1), ("s2", "f2", 1)]
    shown = r"\('s1', 'f2'\), \('s2', 'f1'\)"
    with rejects(f"table: 2 of 4 {NOT_POSITIVE}, e.g. {shown}"):
        load(long(rows))


def test_integer_ids_are_shown_as_integers() -> None:
    rows: Rows = [(1, 10, 1), (1, 20, 0)]
    with rejects(rf"table: 1 of 2 {NOT_POSITIVE}, e.g. \(1, 20\)"):
        load(long(rows, pa.int64(), pa.int64()))


SAMPLES = pa.array(["s1", "s2", "s3"])


def align(samples: pa.Array = SAMPLES, **columns: pa.Array) -> pd.DataFrame:
    table = pa.table({"sample_id": samples, **columns})
    return _tables.sample_metadata({"metadata": table}, "metadata", SAMPLES)


def test_metadata_is_aligned_to_the_tables_samples_and_extras_dropped() -> None:
    frame = align(pa.array(["s3", "x", "s1", "s2"]), age=pa.array([30, 99, 10, 20]))
    assert list(frame.index) == ["s1", "s2", "s3"]
    assert list(frame.columns) == ["age"]
    assert list(frame["age"]) == [10, 20, 30]


def test_metadata_ids_are_normalized_like_the_tables() -> None:
    table = pa.table({"sample_id": pa.array([2, 1], pa.int32()), "g": ["b", "a"]})
    frame = _tables.sample_metadata(
        {"metadata": table}, "metadata", pa.array([1, 2], pa.int64())
    )
    assert list(frame["g"]) == ["a", "b"]


@pytest.mark.parametrize(
    "table",
    [pa.table({"id": ["s1"]}), pa.table([pa.array(["s1"])] * 2, ["sample_id"] * 2)],
    ids=["absent", "repeated"],
)
def test_metadata_has_one_sample_id_column(table: pa.Table) -> None:
    with rejects("metadata: needs exactly one sample_id column"):
        _tables.sample_metadata({"metadata": table}, "metadata", SAMPLES)


def test_covariate_names_are_distinct() -> None:
    table = pa.table(
        [SAMPLES, pa.array([1] * 3), pa.array([2] * 3)], ["sample_id", "a", "a"]
    )
    with rejects("metadata: column 'a' appears twice"):
        _tables.sample_metadata({"metadata": table}, "metadata", SAMPLES)


def test_metadata_ids_are_the_tables_kind() -> None:
    with rejects("metadata: sample_id is int64, the table's is string"):
        align(pa.array([1, 2, 3]))


def test_every_table_sample_has_metadata() -> None:
    with rejects("metadata: 2 of 3 table samples are missing, e.g. 's2', 's3'"):
        align(pa.array(["s1", "x"]))


def test_a_sample_appears_once_in_metadata() -> None:
    with rejects("metadata: 1 of 4 sample_ids repeat, e.g. 's2'"):
        align(pa.array(["s1", "s2", "s3", "s2", "x"]))


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1, 2, 3], pa.date32()),
        pa.array([[1]] * 3),
        pa.array([1, 2, 1]).dictionary_encode(),
    ],
    ids=["date", "list", "integer dictionary"],
)
def test_covariates_have_supported_types(column: pa.Array) -> None:
    kind = str(column.type).replace("[", r"\[").replace("]", r"\]")
    expected = (
        f"metadata: column 'c' is {kind}; covariates are booleans, integers, "
        "floating point, strings, or dictionaries of strings"
    )
    with rejects(expected):
        align(c=column)


@pytest.mark.parametrize(
    ("column", "problem"),
    [
        (pa.array(["a", None, "b"]), "null"),
        (pa.array([1.0, None, 2.0]), "null, NaN or infinite"),
        (pa.array([1.0, float("nan"), 2.0]), "null, NaN or infinite"),
        (pa.array([1.0, float("-inf"), 2.0]), "null, NaN or infinite"),
    ],
    ids=["null", "null float", "nan", "inf"],
)
def test_a_covariate_is_defined_for_every_table_sample(
    column: pa.Array, problem: str
) -> None:
    with rejects(f"metadata: column 'c' is {problem} for 1 of 3 samples, e.g. 's2'"):
        align(c=column)


def test_an_undefined_covariate_on_a_dropped_sample_is_fine() -> None:
    frame = align(pa.array(["s1", "s2", "s3", "x"]), c=pa.array([1.0, 2.0, 3.0, None]))
    assert list(frame["c"]) == [1.0, 2.0, 3.0]


LEVELS = pa.array(["obese", "lean", "overweight"])


def dictionary(indices: list[int], levels: pa.Array = LEVELS) -> pa.Array:
    return pa.DictionaryArray.from_arrays(pa.array(indices, pa.int8()), levels)


def test_a_dictionary_keeps_its_order() -> None:
    frame = align(c=dictionary([1, 2, 0]))
    assert list(frame["c"].cat.categories) == ["obese", "lean", "overweight"]
    assert list(frame["c"]) == ["lean", "overweight", "obese"]


def test_a_level_used_only_by_a_dropped_sample_is_dropped() -> None:
    frame = align(pa.array(["s1", "x", "s2", "s3"]), c=dictionary([0, 2, 1, 0]))
    assert list(frame["c"].cat.categories) == ["obese", "lean"]


def test_every_chunk_carries_the_same_dictionary() -> None:
    levels = pa.array(["lean", "obese", "overweight"])
    chunks = pa.chunked_array([dictionary([0, 1]), dictionary([0], levels)])
    table = pa.table({"sample_id": SAMPLES, "c": chunks})
    with rejects("metadata: column 'c' has a different dictionary in some chunk"):
        _tables.sample_metadata({"metadata": table}, "metadata", SAMPLES)


def test_a_dictionary_holds_each_value_once() -> None:
    with rejects("metadata: column 'c' repeats a dictionary value"):
        align(c=dictionary([0, 1, 0], pa.array(["a", "b", "a"])))


def test_covariates_reach_patsy_as_pandas_would_read_them() -> None:
    frame = align(
        b=pa.array([True, False, True]),
        i=pa.array([1, 2, 3], pa.int32()),
        f=pa.array([0.5, 1.5, 2.5]),
        s=pa.array(["x", "y", "x"], pa.string_view()),
        d=dictionary([0, 1, 2]),
    )
    read = pd.DataFrame({"s": ["x", "y", "x"]})
    assert frame["b"].dtype == bool
    assert pd.api.types.is_integer_dtype(frame["i"].dtype)
    assert frame["f"].dtype == np.float64
    assert frame["s"].dtype == read["s"].dtype
    assert isinstance(frame["d"].dtype, pd.CategoricalDtype)
