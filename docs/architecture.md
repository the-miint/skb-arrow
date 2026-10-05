# Architecture

Why a process host: DESIGN §3.1. Why stateless: §3.3. Why layered: §2.

A caller spawns `skb-arrow` and exchanges JSON lines on stdin/stdout
([`protocol.md`](protocol.md)). Bulk data travels as Arrow IPC in memory-mapped files named by
those messages ([`transport.md`](transport.md)).

## Layers

| Layer | Owns | Must not know |
|---|---|---|
| transport | mmap'd segments, chunking, Arrow IPC encode/decode, unlink-after-open | capabilities, scikit-bio |
| protocol | envelopes, init handshake, versioning, error taxonomy | scikit-bio |
| registry | capability declaration, param validation, schema versions | transport details |
| capabilities | one module per scikit-bio capability, plus shared input contracts | transport, JSON |

A capability takes typed params and named Arrow tables and returns one Arrow table. It never
touches transport or JSON; it raises skb_arrow error classes for faults it anticipates
([`errors.md`](errors.md)).

## Modules

| Module | Holds |
|---|---|
| `cli.py` | entry point: [modes](#cli) |
| `doctor.py` | `versions`: `--version`'s lines; `report`: `--doctor` |
| `host.py` | `reserve`: the channel, the stderr drainer, and the fork hook, before heavy imports; `serve`: DIR checks, the response watch, loop, cleanup |
| `errors.py` | error classes (raised in every layer), `classify`, `collect_warnings` |
| `protocol.py` | `PROTOCOL_VERSION`, `Session`, `handle`: one request line → one response line |
| `transport.py` | segment `read`, `write`, `dispose`, name checks |
| `registry.py` | `Capability`, `CAPABILITIES`, `validate` |
| `capabilities/echo.py` | returns its input |
| `capabilities/_tables.py` | input-table contracts shared by capabilities |
| `capabilities/ancombc.py` | ANCOM-BC, its formula and grouping checks |
| `capabilities/mantel.py` | the Mantel test between two distance tables |

A module becomes a package when it outgrows one file; imports don't change.

## CLI
One mode per invocation:
- `--segment-dir DIR` serves ([`protocol.md`](protocol.md)).
- `--version` prints the package version, `protocol_version`, and each capability with its
  `schema_version`.
- `--doctor` prints `--version`'s lines; a `key: value` line each for `python`, `platform`,
  `dependencies`, `numpy`, `numba`, `threads`, and `segments`; then a `problem:` line per
  fault. It exits 1 if there is one, else 0. Faults:
  - Python other than 3.14; a platform other than Linux x86_64 or macOS arm64.
  - A dependency missing, or installed at a version its pin excludes.
  - [Placement](transport.md): `/dev/shm` on Linux, `$TMPDIR` on macOS (unset is a fault).
    A DIR can't be created there, or it has less free space than one default segment.
  - Anything a check raises, such as the registry failing to import: a `problem:` line,
    never a traceback.

  It compiles nothing: numba's thread count and layer are as configured.
