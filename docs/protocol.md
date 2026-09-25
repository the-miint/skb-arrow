# Protocol

The contract between skb-arrow and its callers. Implementation: `src/skb_arrow/protocol.py`.

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
One UTF-8 JSON object per line, with a string `type`. Lines are handled one at a time; responses
come in request order, so a caller may pipeline. Shutdown: the caller closes stdin.

| `type` | | Fields |
|---|---|---|
| `init` | → | `protocol_version` int; `segment_bytes` int ≥ 1024, optional (default 256 MiB) |
| `ready` | ← | `protocol_version`; `host_version` str; `capabilities` {name: `schema_version`} |
| `call` | → | `id` str or int, optional; `capability` str; `params` object, optional; `input` {table: [segment names]} |
| `result` | ← | `id`; `output` [segment names]; `warnings` |
| `error` | ← | `id` (null if unknown); `kind`; `message`; `warnings`; `traceback` (`internal` only); `protocol_version` (answering `init`) |

- `init` comes first, once. An int is a JSON integer, never `true`; a null field is absent.
- `host_incompatible`, and the host keeps serving: a line that isn't a UTF-8 JSON object, an
  unknown `type`, a missing or mistyped field, `call` before `init`, a second `init`, another
  `protocol_version`, `segment_bytes` below 1024, an unknown capability, or a segment rule
  broken ([`transport.md`](transport.md#segments)).
- `invalid_param`: a param the capability doesn't declare, or input tables other than those it
  declares ([`capabilities.md`](capabilities.md#registry)).
- Every segment a call names is unlinked before its response; `output` is one table.
- `warnings`: [`errors.md`](errors.md#warnings).
