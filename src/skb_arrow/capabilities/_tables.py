"""Input-table contracts shared by capabilities (docs/capabilities.md#input-tables)."""

import bisect
import itertools
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from skb_arrow.errors import InvalidInput

_COLUMNS = ["sample_id", "feature_id", "value"]
_DISTANCE_COLUMNS = ["id_a", "id_b", "distance"]
_SHOWN = 5


@dataclass(frozen=True, eq=False)  # its arrays have no single truth value
class FeatureTable:
    matrix: npt.NDArray[np.float64]  # samples × features, Fortran order
    samples: pa.Array  # sorted, normalized IDs
    features: pa.Array


def feature_table(name: str, table: pa.Table, *, pseudocount: float) -> FeatureTable:
    """Input `name` as a strictly positive dense matrix, once it keeps the contract."""
    _columns(name, table, _COLUMNS)
    sample, feature = (_ids(name, table, column) for column in _COLUMNS[:2])
    value = _numbers(name, table, "value")
    if table.num_rows == 0:
        raise InvalidInput(f"{name}: no rows")
    samples, features = _sorted_unique(sample), _sorted_unique(feature)
    rows = pc.index_in(sample, value_set=samples).to_numpy()
    columns = pc.index_in(feature, value_set=features).to_numpy()
    cells = rows.astype(np.int64) * len(features) + columns

    def pairs(codes: Iterable[int]) -> Iterator[tuple[object, object]]:
        for code in codes:
            r, c = divmod(int(code), len(features))
            yield samples[r].as_py(), features[c].as_py()

    if nulls := value.null_count:
        where = np.sort(cells[value.is_null().to_numpy(zero_copy_only=False)])
        raise _offence(
            name, f"value is null in {nulls} of {table.num_rows} rows", pairs(where)
        )
    codes, counts = np.unique(cells, return_counts=True)
    if (repeats := codes[counts > 1]).size:
        raise _offence(
            name,
            f"{repeats.size} of {codes.size} (sample_id, feature_id) pairs repeat",
            pairs(repeats),
        )
    values = value.to_numpy()
    if (negative := np.sort(cells[values < 0])).size:
        raise _offence(
            name,
            f"{negative.size} of {table.num_rows} values are negative",
            pairs(negative),
        )

    matrix = np.zeros((len(samples), len(features)), order="F")
    matrix[rows, columns] = values
    with np.errstate(over="ignore"):  # a cell overflowing to inf is reported below
        matrix += pseudocount
    bad = ~(np.isfinite(matrix) & (matrix > 0))
    if count := int(bad.sum()):
        raise _offence(
            name,
            f"{count} of {matrix.size} cells are not positive and finite",
            pairs(_true_cells(bad)),
        )
    return FeatureTable(matrix, samples, features)


@dataclass(frozen=True, eq=False)
class DistanceTable:
    condensed: npt.NDArray[np.float64]  # scipy's order: (0, 1), (0, 2), …, (1, 2), …
    ids: pa.Array  # sorted, normalized


def distance_matrix(name: str, table: pa.Table) -> DistanceTable:
    """Input `name` as a condensed distance matrix, once it keeps the contract.

    Arrays are freed as soon as they're used: memory peaks at a few times the matrix.
    """
    _columns(name, table, _DISTANCE_COLUMNS)
    a, b = (_ids(name, table, column) for column in _DISTANCE_COLUMNS[:2])
    if a.type != b.type:
        raise InvalidInput(
            f"{name}: id_a and id_b must be the same kind; got {a.type} and {b.type}"
        )
    distance = _numbers(name, table, "distance")
    rows = table.num_rows
    if rows == 0:
        raise InvalidInput(f"{name}: no rows")
    ids = _sorted_unique(pa.chunked_array([*a.chunks, *b.chunks], a.type))
    i, j = (pc.index_in(column, value_set=ids).to_numpy() for column in (a, b))
    if (selves := np.flatnonzero(i == j)).size:
        raise _offence(
            name,
            f"{selves.size} of {rows} rows pair an ID with itself",
            (ids[k].as_py() for k in np.unique(i[selves])),
        )
    n = len(ids)
    low, high = np.minimum(i, j), np.maximum(i, j)
    del i, j
    codes = _codes(n, low, high)
    del low, high

    def pairs(codes: Iterable[int]) -> Iterator[tuple[object, object]]:
        for code in codes:
            r = bisect.bisect_right(range(n), code, key=lambda r: _start(n, r)) - 1
            yield ids[r].as_py(), ids[int(code) - _start(n, r) + r + 1].as_py()

    if nulls := distance.null_count:
        where = np.sort(codes[distance.is_null().to_numpy(zero_copy_only=False)])
        raise _offence(
            name, f"distance is null in {nulls} of {rows} rows", pairs(where)
        )
    ordered = np.sort(codes)
    if (repeated := ordered[1:] == ordered[:-1]).any():
        repeats = np.unique(ordered[1:][repeated])
        unique = rows - int(repeated.sum())
        raise _offence(name, f"{repeats.size} of {unique} pairs repeat", pairs(repeats))
    total = n * (n - 1) // 2
    if missing := total - rows:  # each row its own pair, so none is beyond `total`
        raise _offence(
            name,
            f"{missing} of {total} pairs are missing",
            pairs(_gaps(ordered, total)),
        )
    del ordered, repeated
    values = distance.to_numpy()
    bad = ~(np.isfinite(values) & (values >= 0))
    if count := int(bad.sum()):
        raise _offence(
            name,
            f"{count} of {rows} distances are negative or not finite",
            pairs(np.sort(codes[bad])),
        )
    condensed = np.empty(total)
    condensed[codes] = values
    return DistanceTable(condensed, ids)


def same_ids(name: str, ids: pa.Array, other: str, expected: pa.Array) -> None:
    """Input `name` holds exactly input `other`'s IDs, of their kind."""
    if ids.type != expected.type:
        raise InvalidInput(f"{name}: IDs are {ids.type}, {other}'s are {expected.type}")
    if len(extra := ids.filter(pc.invert(pc.is_in(ids, value_set=expected)))):
        raise _offence(
            name,
            f"{len(extra)} of {len(ids)} IDs are not in {other}",
            extra[:_SHOWN].to_pylist(),
        )
    if len(missing := expected.filter(pc.invert(pc.is_in(expected, value_set=ids)))):
        raise _offence(
            name,
            f"{len(missing)} of {len(expected)} {other} IDs are missing",
            missing[:_SHOWN].to_pylist(),
        )


def sample_metadata(name: str, table: pa.Table, samples: pa.Array) -> pd.DataFrame:
    """Input `name`'s covariates for `samples`, in their order, indexed by ID."""
    if table.column_names.count("sample_id") != 1:
        raise InvalidInput(
            f"{name}: needs exactly one sample_id column; "
            f"got {_names(table.column_names)}"
        )
    covariates = [c for c in table.column_names if c != "sample_id"]
    if repeated := [c for c, n in Counter(covariates).items() if n > 1]:
        raise InvalidInput(f"{name}: column {repeated[0]!r} appears twice")
    values = pa.table({c: _covariate(name, c, table[c]) for c in covariates})
    ids = _ids(name, table, "sample_id")
    if ids.type != samples.type:
        raise InvalidInput(
            f"{name}: sample_id is {ids.type}, the table's is {samples.type}"
        )
    counts = pc.value_counts(ids)
    if len(counts) < len(ids):
        repeats = pc.filter(
            counts.field("values"), pc.greater(counts.field("counts"), 1)
        )
        raise _offence(
            name,
            f"{len(repeats)} of {len(counts)} sample_ids repeat",
            sorted(repeats.to_pylist()),
        )
    positions = pc.index_in(samples, value_set=ids)
    if missing := positions.null_count:
        raise _offence(
            name,
            f"{missing} of {len(samples)} table samples are missing",
            sorted(samples.filter(positions.is_null()).to_pylist()),
        )
    aligned = values.take(positions)
    frame = aligned.to_pandas()
    frame.index = pd.Index(samples.to_pylist())
    for column in covariates:
        _check_defined(name, column, aligned[column], samples)
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.remove_unused_categories()
    return frame


def _ids(name: str, table: pa.Table, column: str) -> pa.ChunkedArray:
    """`column`'s IDs as int64 or string."""
    ids = table[column]
    if pa.types.is_dictionary(ids.type):
        ids = ids.cast(ids.type.value_type)
    if ids.null_count:
        raise InvalidInput(
            f"{name}: {column} is null in {ids.null_count} of {len(ids)} rows"
        )
    if pa.types.is_integer(ids.type):
        try:
            return ids.cast(pa.int64())
        except pa.ArrowInvalid as e:
            raise InvalidInput(f"{name}: {column} must fit int64") from e
    if _is_string(ids.type):
        return ids.cast(pa.string())
    raise InvalidInput(f"{name}: {column} must be integers or strings, not {ids.type}")


def _is_string(kind: pa.DataType) -> bool:
    return bool(
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or pa.types.is_string_view(kind)
    )


def _sorted_unique(ids: pa.ChunkedArray | pa.Array) -> pa.Array:
    unique = pc.unique(ids)
    return unique.take(pc.sort_indices(unique))


def _covariate(name: str, column: str, values: pa.ChunkedArray) -> pa.ChunkedArray:
    """`column`, checked, with strings as `string`."""
    kind = values.type
    if pa.types.is_dictionary(kind) and _is_string(kind.value_type):
        dictionaries = [chunk.dictionary for chunk in values.chunks]
        if any(not d.equals(dictionaries[0]) for d in dictionaries):
            raise InvalidInput(
                f"{name}: column {column!r} has a different dictionary in some chunk"
            )
        if dictionaries and len(pc.unique(dictionaries[0])) < len(dictionaries[0]):
            raise InvalidInput(f"{name}: column {column!r} repeats a dictionary value")
        if dictionaries and dictionaries[0].null_count:
            # Arrow's null checks see only indices, so this would reach pandas.
            raise InvalidInput(
                f"{name}: column {column!r} has a null in its dictionary"
            )
        return values
    if not (
        pa.types.is_boolean(kind)
        or pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or _is_string(kind)
    ):
        raise InvalidInput(
            f"{name}: column {column!r} is {kind}; covariates are booleans, integers, "
            "floating point, strings, or dictionaries of strings"
        )
    return values.cast(pa.string()) if _is_string(kind) else values


def _check_defined(
    name: str, column: str, values: pa.ChunkedArray, samples: pa.Array
) -> None:
    undefined = values.is_null()
    problem = "null"
    if pa.types.is_floating(values.type):
        undefined = pc.or_(undefined, pc.invert(pc.is_finite(values)).fill_null(True))
        problem = "null, NaN or infinite"
    if count := pc.sum(undefined).as_py():
        raise _offence(
            name,
            f"column {column!r} is {problem} for {count} of {len(samples)} samples",
            sorted(samples.filter(undefined).to_pylist()),
        )


def _columns(name: str, table: pa.Table, columns: list[str]) -> None:
    if sorted(table.column_names) != sorted(columns):
        raise InvalidInput(
            f"{name}: columns must be {', '.join(columns)}; "
            f"got {_names(table.column_names)}"
        )


def _numbers(name: str, table: pa.Table, column: str) -> pa.ChunkedArray:
    values = table[column]
    if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
        raise InvalidInput(
            f"{name}: {column} must be integers or floating point, not {values.type}"
        )
    return values


def _start(n: int, row: int) -> int:
    """Where `row`'s pairs start in the condensed form of `n` IDs."""
    return row * (2 * n - row - 1) // 2


def _codes(
    n: int, low: npt.NDArray[np.int32], high: npt.NDArray[np.int32]
) -> npt.NDArray[np.int64]:
    """Each pair's place in the condensed form: `_start` of `low`, then `high`'s offset.

    In place, so there is one array the size of the rows at a time.
    """
    codes = low.astype(np.int64)
    np.subtract(2 * n - 1, codes, out=codes)
    codes *= low
    codes //= 2
    codes += high
    codes -= low
    codes -= 1
    return codes


def _gaps(present: npt.NDArray[np.int64], total: int) -> Iterator[int]:
    """The codes below `total` that sorted, distinct `present` lacks, in order."""
    yield from range(int(present[0]))
    for k in np.flatnonzero(np.diff(present) > 1):
        yield from range(int(present[k]) + 1, int(present[k + 1]))
    yield from range(int(present[-1]) + 1, total)


def _true_cells(mask: npt.NDArray[np.bool_]) -> Iterator[int]:
    """Codes of `mask`'s true cells, in row order."""
    # A row at a time: np.nonzero would index every true cell at once, 16 bytes each.
    for r, row in enumerate(mask):
        for c in np.flatnonzero(row):
            yield r * mask.shape[1] + int(c)


def _names(columns: Iterable[str]) -> str:
    return ", ".join(repr(c) for c in columns)


def _offence(name: str, problem: str, examples: Iterable[object]) -> InvalidInput:
    shown = ", ".join(repr(e) for e in itertools.islice(examples, _SHOWN))
    return InvalidInput(f"{name}: {problem}, e.g. {shown}")
