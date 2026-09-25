import os
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
from pyarrow import ipc

from skb_arrow.errors import HostIncompatible, InvalidInput

_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")
_OUTPUT = "skbout-"


def check_names(names: Iterable[str]) -> None:
    """Raise `HostIncompatible` unless `names` suit one call's input segments."""
    seen: set[str] = set()
    for name in names:
        if not _NAME.fullmatch(name):
            raise HostIncompatible(f"bad segment name {name!r}")
        folded = name.lower()  # APFS is case-insensitive.
        if folded.startswith(_OUTPUT):
            raise HostIncompatible(f"segment {name!r} uses the reserved {_OUTPUT!r}")
        if folded in seen:
            raise HostIncompatible(f"segment {name!r} named twice")
        seen.add(folded)


def read(directory: Path, names: Sequence[str]) -> pa.Table:
    """The table in segments `names`; all are unlinked, whether or not it succeeds."""
    with ExitStack() as stack:
        try:
            sources = [stack.enter_context(_map(directory, name)) for name in names]
        finally:
            dispose(directory, names)
        tables = [_decode(s, name) for s, name in zip(sources, names, strict=True)]
    for name, table in zip(names, tables, strict=True):
        if not table.schema.equals(tables[0].schema, check_metadata=True):
            raise InvalidInput(f"segment {name}: schema differs from {names[0]}")
    # Not pa.concat_tables: it drops the row count of zero-column tables.
    batches = [batch for table in tables for batch in table.to_batches()]
    return pa.Table.from_batches(batches, tables[0].schema)


def write(
    directory: Path, table: pa.Table, cap: int, counter: Iterator[int]
) -> list[str]:
    """Segment names holding `table`, named from `counter`; a failure leaves none."""
    names: list[str] = []
    try:
        for group in _pack(table.to_batches(), cap):
            name = f"{_OUTPUT}{next(counter)}"
            with _create(directory / name) as file:
                names.append(name)
                with ipc.new_stream(file, table.schema) as writer:
                    for batch in group:
                        writer.write_batch(batch)
    except BaseException:
        dispose(directory, names)
        raise
    return names


def dispose(directory: Path, names: Iterable[str]) -> None:
    """Unlink segments `names`, skipping absent ones and names no segment can have."""
    for name in names:
        if _NAME.fullmatch(name):
            with suppress(FileNotFoundError):
                os.unlink(directory / name)


def _map(directory: Path, name: str) -> pa.MemoryMappedFile:
    try:
        return pa.memory_map(str(directory / name))
    except FileNotFoundError as e:
        raise InvalidInput(f"segment {name} not found") from e


def _decode(source: pa.MemoryMappedFile, name: str) -> pa.Table:
    try:
        table = ipc.open_stream(source).read_all()
        table.validate()
    # pyarrow reports a malformed stream as a bare OSError.
    except (pa.ArrowInvalid, OSError) as e:
        raise InvalidInput(f"segment {name}: {e}") from e
    return table


def _pack(batches: list[pa.RecordBatch], cap: int) -> Iterator[list[pa.RecordBatch]]:
    """Groups of batches, one per segment: greedy, in order, at least one."""
    group: list[pa.RecordBatch] = []
    size = 0
    for batch in (piece for b in batches for piece in _split(b, cap)):
        if group and size + batch.nbytes > cap:
            yield group
            group, size = [], 0
        group.append(batch)
        size += batch.nbytes
    yield group


def _split(batch: pa.RecordBatch, cap: int) -> Iterator[pa.RecordBatch]:
    if batch.nbytes <= cap or batch.num_rows <= 1:
        yield batch
        return
    halves = batch.slice(0, batch.num_rows // 2), batch.slice(batch.num_rows // 2)
    # A shared buffer (a dictionary) counts in full in every slice.
    if 4 * min(half.nbytes for half in halves) > 3 * batch.nbytes:
        yield batch
        return
    for half in halves:
        yield from _split(half, cap)


def _create(path: Path) -> BinaryIO:
    return open(path, "xb")  # Exclusive: never clobber a file the caller left.
