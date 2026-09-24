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

A capability takes typed params and an Arrow table and returns an Arrow table. It never
touches transport, JSON, or error mapping.

## Modules

| Module | Holds |
|---|---|
| `cli.py` | entry point, `--version` |
| `protocol.py` | `PROTOCOL_VERSION` |
| `registry.py` | `CAPABILITIES`: name → `schema_version` |

M2 adds transport and the `echo` capability; M3 adds `ancombc`. A module becomes a package
when it outgrows one file; imports don't change.
