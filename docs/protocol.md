# Protocol

The contract between skb-arrow and its callers. Framing and versioning are decided; message
fields land in M2.

## Channel
- The caller runs `skb-arrow` as a child process.
- Control: one JSON object per line. Requests on stdin, responses on stdout.
- Bulk data never rides the control channel; messages name segments
  ([`transport.md`](transport.md)).
- stderr carries diagnostics only. Discipline rules: M4.
- Every call is a pure function. No sessions, handles, or retained state (DESIGN §3.3).
- No cancel message: the caller kills the process (DESIGN §3.8).

## Versioning (DESIGN §4)
- `protocol_version`: integer, bumped on any wire-format change. The init handshake rejects
  a mismatch.
- The init reply advertises each capability's `schema_version`; a caller may refuse one it
  doesn't understand.
- A caller may require a minimum version and fail fast.
- Unknown request fields are ignored. Unknown error `kind`s map to `internal`
  ([`errors.md`](errors.md)).

Current: `PROTOCOL_VERSION = 1` (`src/skb_arrow/protocol.py`).

## Messages
Init handshake, request and response envelopes: M2.
