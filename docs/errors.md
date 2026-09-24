# Errors and warnings

Why one general mechanism: DESIGN §3.5–3.6. Kinds and rules are fixed. The classifier is the
starting table, finalized with tests in M2; envelope fields land in M2.

Every failure returns one structured envelope carrying a `kind`. Never per-capability.

| `kind` | Cause | Source |
|---|---|---|
| `host_incompatible` | protocol or capability schema drift | handshake |
| `invalid_input` | data fails validation; `LinAlgError` | input pre-flight (M3), classifier |
| `invalid_param` | bad formula, unknown method or param, out-of-range value | registry param validation, classifier |
| `unsupported` | capability needs an extra that isn't installed | classifier |
| `resource` | `MemoryError` | classifier |
| `internal` | anything unclassified — a bug | classifier |

## Classification
One exception-type table for every capability:

| Exception | `kind` |
|---|---|
| `ValueError`, `TypeError`, `LinAlgError` | `invalid_input` |
| `KeyError`, `PatsyError` | `invalid_param` |
| `MemoryError` | `resource` |
| `ImportError` | `unsupported` |
| anything else | `internal` |

scikit-bio reports a bad method or out-of-range value as `ValueError`, so those reach
`invalid_param` only through registry param validation, before scikit-bio runs.

## Rules
1. Callers map an unknown `kind` to `internal`, so a newer host can add kinds without
   breaking an older caller.
2. Only `internal` may carry a traceback; callers must not assume one. Every other kind is
   actionable without one.
3. No `cancelled`: the caller kills the host, so the host cannot report it.

## Warnings
Every capability call records its warnings. Successful responses carry
`warnings: [{category, message, count}]`, deduplicated and capped. The caller decides where
they go.

## Open (M2)
- Host bugs raising `TypeError`, `KeyError`, or `ImportError` (e.g. a mistyped import)
  classify as caller errors with no traceback. Decide how to tell host bugs from bad input.
- `catch_warnings(record=True)` alone keeps the default filters: each source line records
  once, so `count` is wrong, and `DeprecationWarning` is dropped. Counting needs
  `simplefilter("always")`; forwarding deprecations is a decision.
- Whether failure responses carry the warnings recorded before the failure.
