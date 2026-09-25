import os
import re
from collections.abc import Iterable, Iterator, Sequence
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
        folded = _checked(name).lower()  # APFS is case-insensitive.
        if folded in seen:
            raise HostIncompatible(f"segment {name!r} named twice")
        seen.add(folded)


def read(directory: Path, names: Sequence[str]) -> pa.Table:
    """The table in segments `names`; all are unlinked, whether or not it succeeds."""
    try:
        if not names:
            raise HostIncompatible("a table needs at least one segment")
        schema, batches = _load(directory, names[0])
        for name in names[1:]:
            other, more = _load(directory, name)
            if not other.equals(schema, check_metadata=True):
                raise InvalidInput(f"segment {name}: schema differs from {names[0]}")
            batches += more
    finally:
        dispose(directory, names)
    # Not pa.concat_tables: it drops the row count of zero-column tables.
    return pa.Table.from_batches(batches, schema)


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
    """Unlink segments `names`, skipping absent ones and names no segment can have.

    Every name is tried; the first failure is raised after.
    """
    failure: OSError | None = None
    for name in names:
        if _NAME.fullmatch(name):
            try:
                os.unlink(directory / name)
            except FileNotFoundError:
                pass
            except OSError as e:
                failure = failure or e
    if failure is not None:
        raise failure


def _checked(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise HostIncompatible(f"bad segment name {name!r}")
    return name


def _load(directory: Path, name: str) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    try:
        source = pa.memory_map(str(directory / _checked(name)))
    except FileNotFoundError as e:
        raise InvalidInput(f"segment {name} not found") from e
    # Closed once decoded: mapped data outlives the file, and fds stay bounded.
    with source:
        try:
            reader = ipc.open_stream(source)
            batches = list(reader)
            for batch in batches:
                batch.validate()
        except MemoryError:
            raise
        # A malformed stream raises many Arrow types, and bare OSError.
        except (pa.ArrowException, OSError) as e:
            raise InvalidInput(f"segment {name}: {e}") from e
        return reader.schema, batches


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
    sizes = [half.nbytes for half in halves]
    # A shared buffer (a dictionary) counts in full in every slice.
    if max(sizes) > cap and 4 * min(sizes) > 3 * batch.nbytes:
        yield batch
        return
    for half in halves:
        yield from _split(half, cap)


def _create(path: Path) -> BinaryIO:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as e:
        # Only a caller ignoring the reserved prefix can have made it.
        raise HostIncompatible(
            f"segment {path.name} exists; {_OUTPUT!r} is reserved"
        ) from e
    return os.fdopen(fd, "wb")
