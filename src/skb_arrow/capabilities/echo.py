from collections.abc import Mapping

import pyarrow as pa


def run(tables: Mapping[str, pa.Table], params: Mapping[str, object]) -> pa.Table:
    """Its input, unchanged: exercises the machinery."""
    return tables["table"]
