"""Multi-GiB payloads at the default cap: the large workflow runs them (-m large)."""

import itertools
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from client import INIT, SKB_ARROW, Client
from pyarrow import ipc

from skb_arrow import protocol, transport

pytestmark = pytest.mark.large

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30
CAP = protocol._DEFAULT_SEGMENT_BYTES
SCHEMA = pa.schema({"n": pa.int64()})


def counting(total: int, batch: int = 64 * MiB) -> Iterator[pa.RecordBatch]:
    """`total` bytes of int64 counting up from 0, `batch` bytes a batch."""
    values = total // 8
    for start in range(0, values, batch // 8):
        stop = min(start + batch // 8, values)
        yield pa.record_batch([np.arange(start, stop, dtype=np.int64)], schema=SCHEMA)


def assert_counting(table: pa.Table, total: int) -> None:
    seen = 0
    for batch in table.to_batches():
        expected = np.arange(seen, seen + batch.num_rows, dtype=np.int64)
        assert np.array_equal(batch["n"].to_numpy(), expected)
        seen += batch.num_rows
    assert seen == total // 8


@pytest.fixture
def directory() -> Iterator[Path]:
    """An empty DIR, placed as a caller would (docs/transport.md#session-directory)."""
    place = "/dev/shm" if sys.platform == "linux" else tempfile.gettempdir()
    parent = Path(tempfile.mkdtemp(dir=place))
    directory = parent / "session"
    directory.mkdir()
    yield directory
    shutil.rmtree(parent)  # DIR if a test failed, and the client's stderr log


def need(directory: Path, size: int) -> None:
    """Fail, not skip, without room for `size` bytes: these tests exist to run."""
    free = shutil.disk_usage(directory).free
    if free < size:
        pytest.fail(
            f"{size / GiB:.2f} GiB needed in {directory}; {free / GiB:.2f} free"
        )


def test_past_2_gib_round_trips_through_the_host(directory: Path) -> None:
    total = 2 * GiB + 64 * MiB
    need(directory, 2 * total + GiB)  # the input, the output, room to spare
    client = Client(SKB_ARROW, directory, timeout=600)
    try:
        client.send(INIT)
        with open(directory / "in", "xb") as file, ipc.new_stream(file, SCHEMA) as w:
            for batch in counting(total):
                w.write_batch(batch)
        message = {"type": "call", "capability": "echo", "input": {"table": ["in"]}}
        names = client.send(message)["output"]
        assert len(names) > 1
        sizes = [(directory / name).stat().st_size for name in names]
        assert max(sizes) <= CAP + 4 * KiB
        output = client.fetch(names)
        assert_counting(output, total)
        del output
        assert client.shut() == 0
        assert not directory.exists()
        # The input is mapped, not copied, and outputs are written, not mapped.
        assert client.peak_rss < 1.5 * total
    finally:
        client.kill()


def test_one_batch_past_a_gib_splits_within_the_cap(directory: Path) -> None:
    total = 5 * GiB // 4
    need(directory, total + GiB)
    (batch,) = counting(total, batch=total)
    names = transport.write(
        directory, pa.Table.from_batches([batch]), CAP, itertools.count()
    )
    assert len(names) > 1
    for name in names:
        assert os.stat(directory / name).st_size <= CAP + 4 * KiB
    del batch
    assert_counting(transport.read(directory, names), total)
