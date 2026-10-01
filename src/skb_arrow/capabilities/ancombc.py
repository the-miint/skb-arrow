"""ANCOM-BC differential abundance (docs/capabilities.md#ancombc)."""

import keyword
import unicodedata
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from patsy import INTERCEPT, ModelDesc, PatsyError
from skbio.stats.composition import ancombc

from skb_arrow.capabilities import _tables
from skb_arrow.errors import InvalidParam

POSTHOC = ("global", "pairwise", "dunnett")  # the order their rows come in
P_ADJUST = ("holm", "bonferroni", "bh", "by")


def run(tables: Mapping[str, pa.Table], params: Mapping[str, Any]) -> pa.Table:
    counts = _tables.feature_table(
        "table", tables["table"], pseudocount=params["pseudocount"]
    )
    metadata = _tables.sample_metadata("metadata", tables["metadata"], counts.samples)
    terms = _terms(params["formula"], metadata.columns)
    grouping = params["grouping"]
    if grouping is not None:
        _check_grouping(grouping, terms, metadata)
    elif params["posthoc"]:
        raise InvalidParam("ancombc param 'posthoc' needs 'grouping'")
    result = ancombc(
        counts.matrix,
        metadata,
        params["formula"],
        grouping,
        max_iter=params["max_iter"],
        tol=params["tol"],
        alpha=params["alpha"],
        p_adjust=params["p_adjust"],
    )
    tests = {
        "global": result.global_test,
        "pairwise": result.pairwise_test,
        "dunnett": lambda: result.dunnett_test(
            bootstraps=params["bootstraps"], seed=params["seed"]
        ),
    }
    frames = [("main", result.result)]
    frames += [(test, tests[test]()) for test in POSTHOC if test in params["posthoc"]]
    schema = _schema(counts.features.type)
    return pa.concat_tables(
        _rows(test, frame, counts.features, schema) for test, frame in frames
    )


def _terms(formula: str, columns: pd.Index) -> list[Any]:
    """`formula`'s patsy terms, once it names only metadata columns, bare.

    Nothing is evaluated: patsy would run a factor's code as Python (DESIGN §3.11).
    """
    what = "ancombc param 'formula'"
    try:
        description = ModelDesc.from_formula(formula)
    except PatsyError as e:
        raise InvalidParam(f"{what} does not parse: {e}") from e
    if description.lhs_termlist:
        raise InvalidParam(f"{what} must have no left-hand side")
    terms: list[Any] = description.rhs_termlist
    if not terms:
        raise InvalidParam(f"{what} has no terms")
    for code in (factor.code for term in terms for factor in term.factors):
        if not _bare(code):
            raise InvalidParam(f"{what} names {code!r}, not a bare column name")
        if code not in columns:
            raise InvalidParam(f"{what} names {code!r}, not a metadata column")
    return terms


def _bare(code: str) -> bool:
    """Whether Python reads `code` as a lookup of exactly that name."""
    return (
        code.isidentifier()
        and not keyword.iskeyword(code)
        and code != "__debug__"  # a constant
        and unicodedata.normalize("NFKC", code) == code  # as Python normalizes names
    )


def _check_grouping(grouping: str, terms: list[Any], metadata: pd.DataFrame) -> None:
    what = "ancombc param 'grouping'"
    if grouping not in metadata.columns:
        raise InvalidParam(f"{what} names {grouping!r}, not a metadata column")
    kind = metadata[grouping].dtype
    if pd.api.types.is_numeric_dtype(kind) and not pd.api.types.is_bool_dtype(kind):
        raise InvalidParam(f"{what} must be categorical")
    if not any([f.code for f in term.factors] == [grouping] for term in terms):
        raise InvalidParam(f"{what} must be a term of its own in 'formula'")
    if INTERCEPT not in terms:  # without one, groups are tested against zero
        raise InvalidParam(f"{what} needs an intercept in 'formula'")


def _schema(feature_id: pa.DataType) -> pa.Schema:
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


def _rows(
    test: str, frame: pd.DataFrame, features: pa.Array, schema: pa.Schema
) -> pa.Table:
    """`frame`, a scikit-bio result on a bare matrix, as output rows."""
    n = len(frame)
    termed = frame.index.nlevels == 2  # (feature, term); `global` has no term

    def number(column: str) -> pa.Array:
        # From numpy, so NaN stays NaN; from pandas it would become null.
        return pa.array(frame[column].to_numpy(dtype=np.float64))

    absent = pa.nulls(n, pa.float64())
    return pa.table(
        {
            # A bare matrix's features are its column positions.
            "feature_id": features.take(frame.index.get_level_values(0).to_numpy()),
            "test": pa.array([test] * n, pa.string()),
            "term": pa.array(
                frame.index.get_level_values(1).tolist() if termed else [None] * n,
                pa.string(),
            ),
            "lfc": number("Log(FC)") if termed else absent,
            "se": number("SE") if termed else absent,
            "w": number("W"),
            "pvalue": number("pvalue"),
            "qvalue": number("qvalue"),
            "signif": pa.array(frame["Signif"].to_numpy(dtype=bool)),
        },
        schema=schema,
    )
