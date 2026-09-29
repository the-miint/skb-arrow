"""Input-table contracts shared by capabilities (docs/capabilities.md#input-tables)."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from skb_arrow.errors import InvalidInput

_COLUMNS = ["sample_id", "feature_id", "value"]
_SHOWN = 5


@dataclass(frozen=True)
class FeatureTable:
    matrix: npt.NDArray[np.float64]  # samples × features, Fortran order
    samples: pa.Array  # sorted, normalized IDs
    features: pa.Array


def feature_table(
    tables: Mapping[str, pa.Table], name: str, *, pseudocount: float
) -> FeatureTable:
    """Input `name` as a strictly positive dense matrix, once it keeps the contract."""
    table = tables[name]
    if sorted(table.column_names) != sorted(_COLUMNS):
        got = ", ".join(table.column_names)
        raise InvalidInput(f"{name}: columns must be {', '.join(_COLUMNS)}; got {got}")
    sample, feature = (_ids(name, table, column) for column in _COLUMNS[:2])
    value = table["value"]
    if not (pa.types.is_integer(value.type) or pa.types.is_floating(value.type)):
        raise InvalidInput(
            f"{name}: value must be integers or floating point, not {value.type}"
        )
    if table.num_rows == 0:
        raise InvalidInput(f"{name}: no rows")
    samples, features = _sorted_unique(sample), _sorted_unique(feature)
    rows = pc.index_in(sample, value_set=samples).to_numpy()
    columns = pc.index_in(feature, value_set=features).to_numpy()
    cells = rows.astype(np.int64) * len(features) + columns

    def pairs(codes: Iterable[int]) -> list[tuple[object, object]]:
        cells = [divmod(int(code), len(features)) for code in codes]
        return [(samples[r].as_py(), features[c].as_py()) for r, c in cells]

    if nulls := value.null_count:
        where = np.sort(cells[value.is_null().to_numpy(zero_copy_only=False)])
        raise _offence(
            name,
            f"value is null in {nulls} of {table.num_rows} rows",
            pairs(where[:_SHOWN]),
        )
    codes, counts = np.unique(cells, return_counts=True)
    if (repeats := codes[counts > 1]).size:
        raise _offence(
            name,
            f"{repeats.size} of {codes.size} (sample_id, feature_id) pairs repeat",
            pairs(repeats[:_SHOWN]),
        )

    matrix = np.zeros((len(samples), len(features)), order="F")
    matrix[rows, columns] = value.to_numpy()
    matrix += pseudocount
    bad = ~(np.isfinite(matrix) & (matrix > 0))
    if count := int(bad.sum()):
        shown: list[int] = []
        for r in range(len(samples)):  # stops at the first few
            shown += [r * len(features) + int(c) for c in np.flatnonzero(bad[r])]
            if len(shown) >= _SHOWN:
                break
        raise _offence(
            name,
            f"{count} of {matrix.size} cells are not positive and finite",
            pairs(shown[:_SHOWN]),
        )
    return FeatureTable(matrix, samples, features)


def sample_metadata(
    tables: Mapping[str, pa.Table], name: str, samples: pa.Array
) -> pd.DataFrame:
    """Input `name`'s covariates for `samples`, in their order, indexed by ID."""
    table = tables[name]
    if table.column_names.count("sample_id") != 1:
        raise InvalidInput(f"{name}: needs exactly one sample_id column")
    covariates = [c for c in table.column_names if c != "sample_id"]
    for column in covariates:
        if covariates.count(column) > 1:
            raise InvalidInput(f"{name}: column {column!r} appears twice")
    values = pa.table({c: _covariate(name, c, table[c]) for c in covariates})
    ids = _ids(name, table, "sample_id")
    if ids.type != samples.type:
        raise InvalidInput(
            f"{name}: sample_id is {ids.type}, the table's is {samples.type}"
        )
    distinct = _sorted_unique(ids)
    if len(distinct) < len(ids):
        counts = pc.value_counts(ids)
        repeats = pc.filter(
            counts.field("values"), pc.greater(counts.field("counts"), 1)
        )
        raise _offence(
            name,
            f"{len(repeats)} of {len(distinct)} sample_ids repeat",
            _sorted_unique(repeats).to_pylist(),
        )
    positions = pc.index_in(samples, value_set=ids)
    if missing := positions.null_count:
        raise _offence(
            name,
            f"{missing} of {len(samples)} table samples are missing",
            samples.filter(positions.is_null()).to_pylist(),
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
            samples.filter(undefined).to_pylist(),
        )


def _offence(name: str, problem: str, examples: Iterable[object]) -> InvalidInput:
    shown = ", ".join(repr(e) for e in list(examples)[:_SHOWN])
    return InvalidInput(f"{name}: {problem}, e.g. {shown}")
