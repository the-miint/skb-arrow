# Transport

Why memory-mapped files, not POSIX shm: DESIGN §3.2. Why chunked: §3.4. Implementation:
`src/skb_arrow/transport.py`.

## Session directory
- The caller creates a dedicated, empty, private (`mkdtemp`) directory per host and starts
  `skb-arrow --segment-dir DIR`. Every segment, in both directions, is a regular file in DIR.
- The host exits 2 unless DIR exists, is a directory, is readable and writable, and is empty,
  so it never cleans a directory holding anything else. It resolves DIR at start, so a later
  `chdir` or a symlink can't redirect cleanup. The caller writes no segment before `ready`.
- Whoever outlives the other cleans DIR: the host on stdin EOF, a broken pipe, or its
  responses going unread mid-call ([`protocol.md`](protocol.md#channel)), the caller after
  the host exits (`ENOENT` counts as done). A fork child of the host never cleans it. So the
  caller opens every output it needs before closing stdin. Cleanup unlinks DIR's entries,
  then `rmdir`s it, before any diagnostic write (stderr may be a dead pipe).

**Placement:** tmpfs (`/dev/shm`) on Linux, under `$TMPDIR` on macOS. Default Docker containers
mount a 64 MiB `/dev/shm`; raise `--shm-size` or place DIR elsewhere. No fallback: a full
filesystem fails the call with `resource`. `skb-arrow --doctor` checks the placement
([`architecture.md`](architecture.md#cli)).

## Segments
- A table is a non-empty list of segments (else `host_incompatible`), each one uncompressed
  Arrow IPC stream. Its segments share a schema, metadata included, and concatenate in order.
  An empty table is one schema-only segment.
- **Input names** match `[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}` and are distinct within a call,
  compared case-insensitively (APFS). Violations are `host_incompatible`.
- **Output names** are `skbout-<n>`, n counting from 0 per session, each created exclusively,
  mode 0600. The caller never creates one; a file already there is `host_incompatible`. An
  output may be named as an input, chaining calls without a copy.
- A name is single-use until the response to the call naming it arrives.
- **Receiver disposes.** The host unlinks every segment a call names, read or not, before
  responding; the caller unlinks outputs. A read maps each segment, then unlinks it: the data
  stays valid, zero-copy, until the table is freed.
- **Writes** use file I/O, never a mapping, so a full filesystem raises `ENOSPC` (`resource`),
  not `SIGBUS`. A failed write unlinks every segment it created.
- `invalid_input` on read: a missing segment, a malformed stream (any Arrow error but memory
  exhaustion), a schema differing between segments, or a structural fault (Arrow's
  `validate()`).

## Splitting
Why bytes as written, and the default: DESIGN §3.15.

`segment_bytes` (from `init`, default 256 MiB) caps each segment's bytes as written, less its
schema and the dictionaries its first batch carries. A target, not a bound:
- Batches are packed greedily, in order. Each counts the bytes its write adds: its message,
  and any dictionary that differs from the batch before's.
- A batch whose message alone is over the cap is halved until it fits or is one row. It is
  written whole instead when a half is still over the cap and both halves keep more than 3/4
  of its message: a cost every slice pays, such as a view column's data or a wide schema's
  metadata.
- So a segment passes the cap only by its schema, its first batch's dictionaries and the
  8-byte end marker, or by one batch that halving can't shrink.

## Residuals
- Both processes killed at once leave DIR behind.
- A caller killed mid-call leaves the host running until the call leaves any frame that holds
  the GIL, or, with responses to a file, until the call ends.
- A host stopped mid-call skips the capability's own cleanup: temporary files it made and
  children it started may be left behind, as when the caller kills the host.
- The caller is trusted to write valid Arrow: only the structural check runs. A corrupt
  offset or dictionary index that passes it can crash the host. Full validation costs a pass
  over the data (~90 ms/GiB).
- A compressed stream is read, but decompressed onto the heap, not mapped; a corrupt
  compressed length reports `resource`.
- A dictionary is written whole in every segment whose batches use it, so one larger than the
  cap repeats in each.
- A batch that halving can't shrink is written whole, however far past the cap: a view
  column whose data is larger than the cap stays in one segment.
