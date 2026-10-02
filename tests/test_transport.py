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


def written(table: pa.Table, start: int = 0, rows: int | None = None) -> int:
    """The bytes rows `start` on of `table`'s one batch take as a message."""
    (batch,) = table.to_batches()
    return int(ipc.get_record_batch_size(batch.slice(start, rows)))


def assert_within_the_cap(directory: Path, names: list[str], cap: int) -> None:
    """Each segment is within `cap` but for what's exempt, or holds one row."""
    for name in names:
        first, *_ = batches = list(
            ipc.open_stream(pa.memory_map(str(directory / name)))
        )
        if sum(b.num_rows for b in batches) == 1:
            continue
        # The schema, the first batch's dictionaries, the end marker.
        exempt = len(
            stream(pa.Table.from_batches([first]))
        ) - ipc.get_record_batch_size(first)
        assert (directory / name).stat().st_size <= cap + exempt, name


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
    names = write(tmp_path, table, 3 * written(ints(100)))  # three fill it exactly
    assert rows_per_segment(tmp_path, names) == [300, 300, 300, 100]
    assert transport.read(tmp_path, names).equals(table)


def test_an_over_cap_batch_is_halved_until_it_fits(tmp_path: Path) -> None:
    # Halves don't fit, quarters do, but not in pairs.
    names = write(tmp_path, ints(1000), written(ints(1000), 0, 250) * 5 // 4)
    assert rows_per_segment(tmp_path, names) == [250] * 4


# At 501, halving reaches a 2-row piece holding it.
@pytest.mark.parametrize(("at", "rows"), [(500, [500, 1, 499]), (501, [501, 1, 498])])
def test_one_huge_row_is_isolated_and_the_rest_packed(
    tmp_path: Path, at: int, rows: list[int]
) -> None:
    values = ["s"] * 1000
    values[at] = "x" * 100_000
    table = pa.table({"s": values})
    names = write(tmp_path, table, 10_000)
    assert rows_per_segment(tmp_path, names) == rows
    assert transport.read(tmp_path, names).equals(table)


def view_heavy(chars: int) -> pa.Table:
    """1000 rows of `chars` characters in a view column: slices write all its data."""
    strings = pa.array([f"{i:0{chars}d}" for i in range(1000)], pa.string_view())
    return pa.table({"s": strings})


@pytest.mark.parametrize(("chars", "segments"), [(24, 1), (13, 2)])
def test_halving_stops_once_both_halves_keep_over_3_4(
    tmp_path: Path, chars: int, segments: int
) -> None:
    # Halves keep 80%, or 73% and their quarters 81%.
    table = view_heavy(chars)
    names = write(tmp_path, table, 1000)
    assert len(names) == segments
    assert sum((tmp_path / n).stat().st_size for n in names) < 2 * len(stream(table))
    assert transport.read(tmp_path, names).equals(table)


def test_halves_that_fit_are_kept_even_when_they_share_a_buffer(
    tmp_path: Path,
) -> None:
    table = view_heavy(24)  # halves keep 80%
    names = write(tmp_path, table, written(table, 0, 500))
    assert rows_per_segment(tmp_path, names) == [500, 500]


def test_a_cost_every_message_pays_is_not_halved(tmp_path: Path) -> None:
    # 300 columns' metadata alone is over the cap, and each buffer pads to 8 bytes.
    batch = pa.record_batch({f"c{i}": pa.array([1] * 8, pa.int8()) for i in range(300)})
    table = pa.Table.from_batches([batch] * 10)
    names = write(tmp_path, table, 8192)
    assert rows_per_segment(tmp_path, names) == [8] * 10
    assert transport.read(tmp_path, names).equals(table)


def shared_dictionary(dictionary_bytes: int, batches: int = 100) -> pa.Table:
    """`batches` batches of 100 rows, all indexing one `dictionary_bytes` dictionary."""
    dictionary = pa.array([f"{i:0{dictionary_bytes // 100}d}" for i in range(100)])
    column = pa.DictionaryArray.from_arrays(
        pa.array(range(100), pa.int32()), dictionary
    )
    return pa.Table.from_batches(
        [pa.record_batch({"d": column})] * batches, pa.schema({"d": column.type})
    )


def test_batches_sharing_a_dictionary_pack_as_written(tmp_path: Path) -> None:
    # The dictionary is near the cap's size, but written once per segment, and exempt.
    table, cap = shared_dictionary(23_000), 24_000
    names = write(tmp_path, table, cap)
    per_segment = cap // written(table.slice(0, 100).combine_chunks())
    assert len(names) == -(-100 // per_segment)  # not one segment a batch
    assert_within_the_cap(tmp_path, names, cap)
    assert transport.read(tmp_path, names).equals(table)


def test_a_dictionary_over_the_cap_rides_in_every_segment(tmp_path: Path) -> None:
    table, cap = shared_dictionary(50_000, batches=200), 24_000
    names = write(tmp_path, table, cap)
    assert len(names) > 1
    # Not halved: the dictionary takes their nbytes over the cap, not their messages.
    for name in names:
        reader = ipc.open_stream(pa.memory_map(str(tmp_path / name)))
        assert {batch.num_rows for batch in reader} == {100}
    assert_within_the_cap(tmp_path, names, cap)
    assert transport.read(tmp_path, names).equals(table)


def test_a_dictionary_changing_every_batch_counts(tmp_path: Path) -> None:
    def batch(i: int) -> pa.RecordBatch:
        dictionary = pa.array([str(i) * 10_000])  # a different one each batch
        indices = pa.array([0] * 10, pa.int32())
        return pa.record_batch(
            {"d": pa.DictionaryArray.from_arrays(indices, dictionary)}
        )

    batches = [batch(i) for i in range(1, 10)]
    first = pa.Table.from_batches(batches[:1])
    message = ipc.get_record_batch_size(batches[0])
    schema_only = len(stream(first.schema.empty_table()))
    dictionary = len(stream(first)) - schema_only - message
    # The first batch's dictionary is exempt; the next two each add theirs.
    cap = message + 2 * (dictionary + message)
    table = pa.Table.from_batches(batches)
    names = write(tmp_path, table, cap)
    assert rows_per_segment(tmp_path, names) == [30, 30, 30]
    assert transport.read(tmp_path, names).equals(table)


def test_nested_dictionaries_round_trip_across_segments(tmp_path: Path) -> None:
    words = pa.array(["a", "b", "c"] * 100).dictionary_encode()
    table = pa.table(
        {
            "listed": pa.ListArray.from_arrays(pa.array(range(0, 301, 3)), words),
            "nested": pa.StructArray.from_arrays(
                [pa.array(["x", "y", "z"] * 100).dictionary_encode()], ["w"]
            ).slice(0, 100),
        }
    )
    names = write(tmp_path, table, 500)
    assert len(names) > 1
    assert transport.read(tmp_path, names).equals(table)


def test_every_segment_is_within_the_cap_but_its_exemptions(tmp_path: Path) -> None:
    table = pa.concat_tables([SAMPLE] * 200).combine_chunks()
    names = write(tmp_path, table, 2000)
    assert len(names) > 1
    assert_within_the_cap(tmp_path, names, 2000)


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
    names = write(tmp_path, ints(6400), written(ints(100)))
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
