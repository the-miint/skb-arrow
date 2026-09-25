import errno
import gc
import itertools
import os
import resource
import struct
import sys
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pytest
from pyarrow import ipc

from skb_arrow import transport
from skb_arrow.errors import HostIncompatible, InvalidInput

SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), metadata={"unit": "count"}),
        pa.field("name", pa.string()),
        pa.field("tags", pa.list_(pa.int64())),
        pa.field("kind", pa.dictionary(pa.int32(), pa.string())),
        pa.field("when", pa.timestamp("us", tz="UTC")),
    ],
    metadata={"source": "test"},
)
SAMPLE = pa.table(
    {
        "id": [1, None, 3],
        "name": ["a", None, "ccc"],
        "tags": [[1], [], None],
        "kind": ["x", "y", "x"],
        "when": [0, 1, None],
    },
    schema=SCHEMA,
)
UNCAPPED = 1 << 40


def write(directory: Path, table: pa.Table, cap: int = UNCAPPED) -> list[str]:
    return transport.write(directory, table, cap, itertools.count())


def ints(rows: int) -> pa.Table:
    """16 bytes a row, one batch."""
    column = pa.array(range(rows), pa.int64())
    return pa.table({"a": column, "b": column})


def stream(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue())


def rows_per_segment(directory: Path, names: list[str]) -> list[int]:
    """Read without consuming the segments."""
    return [
        ipc.open_stream(pa.memory_map(str(directory / name))).read_all().num_rows
        for name in names
    ]


def test_round_trip_is_exact_including_metadata(tmp_path: Path) -> None:
    result = transport.read(tmp_path, write(tmp_path, SAMPLE))
    assert result.equals(SAMPLE, check_metadata=True)


def test_outputs_are_named_from_the_session_counter(tmp_path: Path) -> None:
    counter = itertools.count()
    first = transport.write(tmp_path, SAMPLE, UNCAPPED, counter)
    second = transport.write(tmp_path, SAMPLE, UNCAPPED, counter)
    assert (first, second) == (["skbout-0"], ["skbout-1"])
    assert sorted(p.name for p in tmp_path.iterdir()) == first + second


def test_batches_pack_greedily_up_to_the_cap(tmp_path: Path) -> None:
    table = pa.Table.from_batches(ints(1000).to_batches(max_chunksize=100))
    # 1600 bytes a batch: three fill 4800 exactly.
    names = write(tmp_path, table, 4800)
    assert rows_per_segment(tmp_path, names) == [300, 300, 300, 100]
    assert transport.read(tmp_path, names).equals(table)


def test_an_over_cap_batch_is_halved_until_it_fits(tmp_path: Path) -> None:
    # 16000 bytes: halves of 8000 don't fit, quarters of 4000 do, but not in pairs.
    names = write(tmp_path, ints(1000), 5000)
    assert rows_per_segment(tmp_path, names) == [250] * 4


def test_one_huge_row_is_isolated_and_the_rest_packed(tmp_path: Path) -> None:
    values = ["s"] * 1000
    values[500] = "x" * 100_000
    table = pa.table({"s": values})
    names = write(tmp_path, table, 10_000)
    assert rows_per_segment(tmp_path, names) == [500, 1, 499]
    assert transport.read(tmp_path, names).equals(table)


def dictionary_heavy(dictionary_bytes: int) -> pa.Table:
    """1000 rows of 12 bytes, and a dictionary every slice counts in full."""
    indices = pa.array([0] * 1000, pa.int32())
    dictionary = pa.array(["d" * dictionary_bytes])
    return pa.table(
        {
            "d": pa.DictionaryArray.from_arrays(indices, dictionary),
            "v": pa.array(range(1000), pa.int64()),
        }
    )


@pytest.mark.parametrize(("dictionary_bytes", "segments"), [(18_000, 1), (8_000, 2)])
def test_halving_stops_once_both_halves_keep_over_75_percent(
    tmp_path: Path, dictionary_bytes: int, segments: int
) -> None:
    # Halves keep 80% or 70% of the batch.
    table = dictionary_heavy(dictionary_bytes)
    names = write(tmp_path, table, 1000)
    assert len(names) == segments
    assert transport.read(tmp_path, names).equals(table)


def test_halves_that_fit_are_kept_even_when_they_share_a_buffer(
    tmp_path: Path,
) -> None:
    table = dictionary_heavy(14_000)  # halves keep 77%
    names = write(tmp_path, table, table.slice(0, 500).nbytes)
    assert rows_per_segment(tmp_path, names) == [500, 500]


def test_a_failed_read_holds_no_segment_open(tmp_path: Path) -> None:
    (tmp_path / "ok").write_bytes(stream(SAMPLE))
    (tmp_path / "bad").write_bytes(b"")
    before = len(os.listdir("/dev/fd"))
    with pytest.raises(InvalidInput) as raised:
        transport.read(tmp_path, ["ok", "bad"])
    # `raised` keeps the failed read's frame, and with it any unclosed mapping.
    assert len(os.listdir("/dev/fd")) == before
    del raised


def test_a_read_holds_one_segment_open_at_a_time(tmp_path: Path) -> None:
    names = write(tmp_path, ints(6400), 1600)
    assert len(names) == 64
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (len(os.listdir("/dev/fd")) + 8, hard))
    try:
        result = transport.read(tmp_path, names)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert result.equals(ints(6400))


def test_zero_column_tables_keep_their_row_count(tmp_path: Path) -> None:
    table = pa.table({"a": [1, 2, 3]}).drop_columns(["a"])
    assert transport.read(tmp_path, write(tmp_path, table)).num_rows == 3


def test_an_empty_table_is_one_schema_only_segment(tmp_path: Path) -> None:
    empty = SCHEMA.empty_table()
    names = write(tmp_path, empty)
    assert len(names) == 1
    assert transport.read(tmp_path, names).equals(empty, check_metadata=True)


def test_reads_are_zero_copy_and_outlive_their_segments(tmp_path: Path) -> None:
    table = ints(1_000_000)
    names = write(tmp_path, table)
    before = pa.total_allocated_bytes()
    result = transport.read(tmp_path, names)
    assert pa.total_allocated_bytes() - before < table.nbytes // 100
    assert list(tmp_path.iterdir()) == []
    gc.collect()
    assert result.equals(table)


@pytest.mark.parametrize(
    "other",
    [
        pa.table({"b": [1]}),
        pa.table({"a": [1]}).replace_schema_metadata({"k": "v"}),
    ],
    ids=["columns", "metadata"],
)
def test_segments_with_differing_schemas_are_invalid_input(
    tmp_path: Path, other: pa.Table
) -> None:
    (tmp_path / "first").write_bytes(stream(pa.table({"a": [1]})))
    (tmp_path / "second").write_bytes(stream(other))
    with pytest.raises(InvalidInput, match="^segment second: "):
        transport.read(tmp_path, ["first", "second"])
    assert list(tmp_path.iterdir()) == []


def replace_once(data: bytes, old: bytes, new: bytes) -> bytes:
    assert data.count(old) == 1
    return data.replace(old, new)


def patch_schema(table: pa.Table, old: bytes, new: bytes) -> bytes:
    """`table`'s stream with one value in its schema message replaced."""
    raw = stream(table)
    end = 8 + struct.unpack_from("<i", raw, 4)[0]  # continuation, length, flatbuffer
    return replace_once(raw[:end], old, new) + raw[end:]


def pack(fmt: str, *values: int) -> bytes:
    return struct.pack(f"<{fmt}", *values)


TWO_DICTIONARIES = pa.table(
    {"x": pa.array(["p"]).dictionary_encode(), "y": pa.array(["q"]).dictionary_encode()}
)
CORRUPT = {
    "empty": b"",
    "garbage": b"\xff" * 64,
    "truncated": stream(ints(1000))[:-600],
    # Structurally invalid: offsets run past the data buffer.
    "overrun offsets": replace_once(
        stream(pa.table({"s": ["hello", ""]})),
        pack("3i", 0, 5, 5),
        pack("3i", 0, 5, 10_000_000),
    ),
    # ArrowNotImplementedError: a 4-bit integer.
    "bit width": patch_schema(pa.table({"a": [1]}), pack("i", 64), pack("i", 4)),
    # ArrowKeyError: the second dictionary batch matches no field's id.
    "dictionary id": patch_schema(TWO_DICTIONARIES, pack("q", 1), pack("q", 7)),
}


@pytest.mark.parametrize("data", CORRUPT.values(), ids=CORRUPT.keys())
def test_a_corrupt_segment_is_invalid_input(tmp_path: Path, data: bytes) -> None:
    (tmp_path / "in").write_bytes(data)
    with pytest.raises(InvalidInput, match="^segment in: "):
        transport.read(tmp_path, ["in"])
    assert list(tmp_path.iterdir()) == []


def test_memory_exhaustion_while_decoding_stays_a_memory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhausted(source: pa.MemoryMappedFile) -> None:
        raise pa.ArrowMemoryError("malloc failed")

    (tmp_path / "in").write_bytes(stream(SAMPLE))
    monkeypatch.setattr(ipc, "open_stream", exhausted)
    with pytest.raises(MemoryError):
        transport.read(tmp_path, ["in"])


def test_a_missing_segment_is_invalid_input_and_unopened_ones_are_disposed(
    tmp_path: Path,
) -> None:
    for name in ["a", "c"]:
        (tmp_path / name).write_bytes(stream(SAMPLE))
    with pytest.raises(InvalidInput, match="^segment b not found$"):
        transport.read(tmp_path, ["a", "b", "c"])
    assert list(tmp_path.iterdir()) == []


def test_a_table_needs_at_least_one_segment(tmp_path: Path) -> None:
    with pytest.raises(HostIncompatible, match="at least one segment"):
        transport.read(tmp_path, [])


def test_read_refuses_names_that_leave_the_directory(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    (tmp_path / "outside").write_bytes(stream(SAMPLE))
    (directory / "a").write_bytes(stream(SAMPLE))
    with pytest.raises(HostIncompatible, match="^bad segment name"):
        transport.read(directory, ["a", "../outside"])
    assert list(directory.iterdir()) == []
    assert (tmp_path / "outside").exists()


def test_a_segment_that_cannot_be_unlinked_does_not_stop_the_rest(
    tmp_path: Path,
) -> None:
    for name in ["a", "c"]:
        (tmp_path / name).touch()
    (tmp_path / "sub").mkdir()
    with pytest.raises(OSError):
        transport.dispose(tmp_path, ["a", "sub", "c"])
    assert [p.name for p in tmp_path.iterdir()] == ["sub"]


@pytest.mark.parametrize(
    "name", ["", ".hidden", "..", "a/b", "a b", "é", "a\n", "a" * 129]
)
def test_malformed_names_are_rejected(name: str) -> None:
    with pytest.raises(HostIncompatible, match="^bad segment name"):
        transport.check_names([name])


def test_names_must_be_distinct_ignoring_case() -> None:
    with pytest.raises(HostIncompatible, match="'seG' named twice"):
        transport.check_names(["Seg", "other", "seG"])


def test_well_formed_distinct_names_pass() -> None:
    transport.check_names(["a", "-", "_x.y-z", "skbout-0", "A" * 128, "b"])


def test_dispose_unlinks_only_names_that_can_be_segments(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    (tmp_path / "outside").touch()
    for name in ["x", "skbout-0"]:
        (directory / name).touch()
    transport.dispose(directory, ["../outside", "x", "skbout-0", "absent"])
    assert list(directory.iterdir()) == []
    assert (tmp_path / "outside").exists()


def test_a_failed_write_unlinks_its_segments_and_spares_others(tmp_path: Path) -> None:
    (tmp_path / "skbout-1").write_bytes(b"not ours")
    with pytest.raises(HostIncompatible, match="^segment skbout-1 exists"):
        write(tmp_path, ints(1000), 5000)
    assert [p.name for p in tmp_path.iterdir()] == ["skbout-1"]
    assert (tmp_path / "skbout-1").read_bytes() == b"not ours"


def test_outputs_are_private_to_their_owner(tmp_path: Path) -> None:
    umask = os.umask(0)
    try:
        names = write(tmp_path, SAMPLE)
    finally:
        os.umask(umask)
    assert (tmp_path / names[0]).stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform != "linux", reason="needs /dev/full")
def test_a_full_filesystem_raises_enospc_and_leaves_no_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create = transport._create

    def full(path: Path) -> BinaryIO:
        file = create(path)
        device = os.open("/dev/full", os.O_WRONLY)
        os.dup2(device, file.fileno())
        os.close(device)
        return file

    monkeypatch.setattr(transport, "_create", full)
    with pytest.raises(OSError) as raised:
        write(tmp_path, ints(100_000))
    assert raised.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []
