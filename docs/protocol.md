# Protocol

The contract between skb-arrow and its callers. Framing and versioning are decided; message
fields land in M2.

## Channel
- The caller runs `skb-arrow` as a child process.
- Control: one JSON object per line. Requests on stdin, responses on stdout.
- stdout carries protocol messages only. The host sends all other output — library prints,
  C-level writes to fd 1 — to stderr. Mechanism: M2.
- stderr carries diagnostics only. Discipline rules: M4.
- Bulk data never rides the control channel; messages name segments
  ([`transport.md`](transport.md)).
- Stateless (DESIGN §3.3). No cancel message: the caller kills the process (DESIGN §3.8).

## Versioning
Why: DESIGN §4.
- `protocol_version`: integer, bumped on any wire-format change. The init handshake rejects
  a mismatch with `host_incompatible`.
- The init reply advertises each capability's `schema_version`; a caller may refuse one it
  doesn't understand.
- A caller may require a minimum version and fail fast with an actionable message.
- Both sides ignore unknown envelope fields.
- Unknown capability params are `invalid_param`: a misspelled `seed` must fail, not fall back
  to the default.
- Unknown error `kind`s map to `internal` ([`errors.md`](errors.md)).

Current: `PROTOCOL_VERSION = 1` (`src/skb_arrow/protocol.py`).

## Messages
Init handshake, request and response envelopes: M2.
