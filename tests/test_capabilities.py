"""Contracts every registered capability keeps (docs/capabilities.md#contract)."""

import os
import pickle
import random
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import skbio

from skb_arrow import registry

_SAMPLES = [f"s{i}" for i in range(12)]
_COUNTS = np.random.default_rng(0).integers(1, 50, size=(12, 5))
_PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def _distances(seed: int) -> pa.Table:
    values = np.random.default_rng(seed).random(len(_PAIRS))
    return pa.table(
        {
            "id_a": [a for a, _ in _PAIRS],
            "id_b": [b for _, b in _PAIRS],
            "distance": values,
        }
    )


# One call per capability, through its stochastic paths where it has them.
EXAMPLES: dict[str, tuple[dict[str, pa.Table], dict[str, Any]]] = {
    "echo": ({"table": pa.table({"a": [1, 2]})}, {}),
    "ancombc": (
        {
            "table": pa.table(
                {
                    "sample_id": np.repeat(_SAMPLES, 5).tolist(),
                    "feature_id": [f"f{j}" for j in range(5)] * 12,
                    "value": _COUNTS.ravel().tolist(),
                }
            ),
            "metadata": pa.table({"sample_id": _SAMPLES, "g": ["a", "b", "c"] * 4}),
        },
        {
            "formula": "g",
            "grouping": "g",
            "posthoc": ["global", "pairwise", "dunnett"],
            "bootstraps": 10,
        },
    ),
    "mantel": ({"x": _distances(1), "y": _distances(2)}, {"permutations": 9}),
}


def pandas_options(node: object = pd.options, prefix: str = "") -> dict[str, object]:
    """Every pandas option's value, by name."""
    options: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reading a deprecated one warns
        for name in dir(node):
            value = getattr(node, name)
            if isinstance(value, type(pd.options)):
                options |= pandas_options(value, f"{prefix}{name}.")
            else:
                options[f"{prefix}{name}"] = value
    return options


def state() -> dict[str, object]:
    return {
        "random": random.getstate(),
        "numpy random": pickle.dumps(np.random.get_state()),
        "scikit-bio config": skbio.get_config(),
        "pandas options": pandas_options(),
        "cwd": os.getcwd(),
        "environ": dict(os.environ),
        "numpy errors": np.geterr(),
    }


def test_every_capability_has_an_example() -> None:
    assert EXAMPLES.keys() == registry.CAPABILITIES.keys()


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_capabilities_leave_process_state_as_found(name: str) -> None:
    tables, params = EXAMPLES[name]
    capability, resolved = registry.validate(name, tables.keys(), params)
    # Fresh from the OS: a capability seeding them shows even with a seed used before.
    random.seed()
    np.random.seed()
    before = state()
    capability.run(tables, resolved)
    assert state() == before
