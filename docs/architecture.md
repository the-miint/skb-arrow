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
| capabilities | one module per scikit-bio capability | transport, JSON |

A capability takes typed params and named Arrow tables and returns one Arrow table. It never
touches transport or JSON; it raises skb_arrow error classes for faults it anticipates
([`errors.md`](errors.md)).

## Modules

| Module | Holds |
|---|---|
| `cli.py` | entry point, `--version` |
| `errors.py` | error classes (raised in every layer), `classify`, `collect_warnings` |
| `protocol.py` | `PROTOCOL_VERSION`, `Session`, `handle`: one request line → one response line |
| `transport.py` | segment `read`, `write`, `dispose`, name checks |
| `registry.py` | `Capability`, `CAPABILITIES`, `validate` |
| `capabilities/echo.py` | returns its input |

M2 adds the host (serving loop, fd reservation); M3 adds `ancombc`. A module becomes a package
when it outgrows one file; imports don't change.
