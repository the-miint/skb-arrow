"""The Mantel test between two distance tables (docs/capabilities.md#mantel)."""

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from skbio import DistanceMatrix
from skbio.stats.distance import mantel

from skb_arrow.capabilities import _tables

METHODS = ("pearson", "spearman", "kendalltau")
ALTERNATIVES = ("two-sided", "greater", "less")
_SCHEMA = pa.schema(
    [
        pa.field("statistic", pa.float64(), nullable=False),
        pa.field("pvalue", pa.float64(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
    ]
)


def run(tables: Mapping[str, pa.Table], params: Mapping[str, Any]) -> pa.Table:
    x, y = (_tables.distance_matrix(name, tables[name]) for name in ("x", "y"))
    _tables.same_ids("y", y.ids, "x", x.ids)
    # Both in sorted ID order: scikit-bio's default IDs then match them as they are.
    statistic, pvalue, n = mantel(
        DistanceMatrix(x.condensed, condensed=True),
        DistanceMatrix(y.condensed, condensed=True),
        method=params["method"],
        permutations=params["permutations"],
        alternative=params["alternative"],
        seed=params["seed"],
        engine="numba",  # DESIGN §3.9
    )
    return pa.table(
        {"statistic": [statistic], "pvalue": [pvalue], "n": [n]}, schema=_SCHEMA
    )
