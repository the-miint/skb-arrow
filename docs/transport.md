# Transport

Why memory-mapped files, not POSIX shm: DESIGN §3.2. Why chunked: §3.4. Implementation:
`src/skb_arrow/transport.py`.

## Session directory
- The caller creates a dedicated, empty directory per host and starts
  `skb-arrow --segment-dir DIR`. Every segment, in both directions, is a regular file in DIR.
- The host exits 2 unless DIR exists, is a directory, is empty, and is writable, so it never
  cleans a directory holding anything else. The caller writes no segment before `ready`.
- Whoever outlives the other cleans DIR: the host on stdin EOF or a broken pipe, the caller
  after the host exits (`ENOENT` counts as done). Cleanup unlinks regular files, then
  `rmdir`s, before any diagnostic write (stderr may be a dead pipe).

**Placement:** tmpfs (`/dev/shm`) on Linux, under `$TMPDIR` on macOS. Default Docker containers
mount a 64 MiB `/dev/shm`; raise `--shm-size` or place DIR elsewhere. No fallback: a full
filesystem fails the call with `resource`.

## Segments
- A table is a non-empty list of segments, each one Arrow IPC stream. Its segments share a
  schema, metadata included, and concatenate in order. An empty table is one schema-only
  segment.
- **Input names** match `[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}`, don't start with `skbout-`, and
  are distinct within a call, compared case-insensitively (APFS). Violations are
  `host_incompatible`.
- **Output names** are `skbout-<n>`, n counting from 0 per session, each created exclusively.
- A name is single-use until the response to the call naming it arrives.
- **Receiver disposes.** The host unlinks every segment a call names, read or not, before
  responding; the caller unlinks outputs. A read maps each segment, then unlinks it: the data
  stays valid, zero-copy, until the table is freed.
- **Writes** use file I/O, never a mapping, so a full filesystem raises `ENOSPC` (`resource`),
  not `SIGBUS`. A failed write unlinks every segment it created.
- `invalid_input` on read: a missing segment, a malformed stream, a schema differing between
  segments, or a structural fault (Arrow's `validate()`).

## Splitting
`segment_bytes` (from `init`, default 256 MiB) caps each segment's sum of batch `nbytes`, IPC
framing excluded. A target, not a bound:
- Batches are packed greedily, in order.
- An over-cap batch is halved until it fits, is one row, or both halves keep more than 75% of
  its `nbytes` (a shared buffer, such as a dictionary); then it is written whole.

## Residuals
- Both processes killed at once leave DIR behind.
- A caller killed mid-call leaves the host running until the call ends (parent-death watchdog:
  M4).
- Content is not validated (`validate(full=True)` costs a pass over the data, ~90 ms/GiB): a
  corrupt offset or dictionary index that passes the structural check can crash the host.

## Open (M4)
- The default cap and the chunk-size policy.
