"""Contracts every registered capability keeps (docs/capabilities.md#contract)."""

import json
import os
import pickle
import random
import tomllib
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import skbio

from skb_arrow import registry

ROOT = Path(__file__).parents[1]
INTERFACES = ROOT / "tests" / "data" / "interfaces.json"
# The libraries that compute answers (docs/capabilities.md#versioning).
COMPUTE = (
    "llvmlite",
    "numba",
    "numpy",
    "pandas",
    "patsy",
    "pyarrow",
    "scikit-bio",
    "scipy",
)
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


# One call per capability, through its stochastic paths where it has them. Their
# outputs' schemas are recorded (interfaces.json): an output type that follows an
# input's follows the type given here.
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


def interface(name: str) -> dict[str, Any]:
    """What `name` takes and gives, as interfaces.json records it."""
    capability = registry.CAPABILITIES[name]
    tables, params = EXAMPLES[name]
    _, resolved = registry.validate(name, tables.keys(), params)
    return {
        "inputs": sorted(capability.inputs),
        "params": [
            {
                "name": key,
                "kind": param.kind.__name__,
                "item": None if param.item is None else param.item.__name__,
                "rule": param.rule,
            }
            | (
                {}
                if param.default is registry._REQUIRED
                else {"default": param.default}
            )
            for key, param in capability.params.items()
        ],
        "output": capability.run(tables, resolved)
        .schema.to_string(truncate_metadata=False)
        .splitlines(),
    }


def records() -> dict[str, Any]:
    """interfaces.json, each key once: json.loads would keep a repeated key's last."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(key for key, _ in pairs)
        repeated = [key for key, count in counts.items() if count > 1]
        assert not repeated, f"repeated: {repeated}"
        return dict(pairs)

    loaded: dict[str, Any] = json.loads(
        INTERFACES.read_text(), object_pairs_hook=unique
    )
    return loaded


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_each_interface_is_its_versions_record(name: str) -> None:
    key = f"{name}/{registry.CAPABILITIES[name].schema_version}"
    recorded, current = records(), interface(name)
    paste = json.dumps({key: current}, indent=2, sort_keys=True)
    assert key in recorded, f"{key} has no record; a bump adds:\n{paste}"
    assert recorded[key] == current, f"{name} changed: bump its version"


def test_records_are_one_per_version_of_each_capability() -> None:
    versions: dict[str, set[int]] = {}
    for key in records():
        name, version = key.rsplit("/", 1)
        versions.setdefault(name, set()).add(int(version))
    assert versions == {
        name: set(range(1, capability.schema_version + 1))
        for name, capability in registry.CAPABILITIES.items()
    }


def test_every_runtime_dependency_is_pinned_to_its_locked_version() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    names = [p["name"] for p in lock["package"]]
    packages = {p["name"]: p for p in lock["package"]}
    closure: set[str] = set()
    markers: set[str | None] = set()
    todo = list(COMPUTE)
    while todo:
        if (name := todo.pop()) in closure:
            continue
        closure.add(name)
        for edge in packages[name].get("dependencies", []):
            assert "extra" not in edge, (
                f"{name} needs {edge['name']}'s extras: walk them"
            )
            markers.add(edge.get("marker"))
            todo.append(edge["name"])
    # The environments' own marker, on every edge: another is a conditional dependency.
    assert len(markers) == 1, f"conditional dependencies: {markers}"
    forked = sorted(n for n in closure if names.count(n) > 1)
    assert not forked, f"locked at two versions, so pin one: {forked}"
    pins = sorted(f"{n}=={packages[n]['version']}" for n in closure)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    block = "".join(f'    "{pin}",\n' for pin in pins)
    assert sorted(project["dependencies"]) == pins, f"dependencies = [\n{block}]"
