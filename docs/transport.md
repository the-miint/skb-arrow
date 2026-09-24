# Transport

Why memory-mapped files, not POSIX shm: DESIGN §3.2. Why chunked: §3.4.

- A payload is a **list of segments**. Each segment is a regular file holding an Arrow IPC
  stream of one or more record batches.
- Segments are memory-mapped and read zero-copy.
- **Unlink after open.** The consumer unlinks a segment once it has it open; the kernel frees
  it on last close. No cleanup registry.
- Placement: `/dev/shm` on Linux, `$TMPDIR` on macOS; overridable.
- A policy cap bounds segment size, keeping each segment clear of the macOS cold-read cliff
  (DESIGN §3.2).

## Open
- **Unopened segments leak (M2).** Unlink-after-open reclaims only segments their consumer
  opened. A segment is orphaned if its producer is killed before handoff — the host killed
  after writing response segments, before the caller reads their names — or if a request is
  rejected before its segments are opened. M2's orphan-free criterion depends on closing this.
- **Small `/dev/shm` (M2).** Default Docker containers mount 64 MiB; large payloads fail with
  `ENOSPC`, or `SIGBUS` through a mapping. Needs detection or fallback, not only the override.
- Segment naming and the placement override (M2); the size cap (M4).
