# Errors and warnings

Why one general mechanism: DESIGN §3.5–3.6. Kinds and rules are fixed; envelope fields land
in M2.

Every failure returns one structured envelope carrying a `kind`. Never per-capability.

| `kind` | Cause | Traceback |
|---|---|---|
| `host_incompatible` | protocol or capability schema drift | no |
| `invalid_input` | data fails scikit-bio validation; `LinAlgError` | no |
| `invalid_param` | bad formula, unknown method, out-of-range value | no |
| `unsupported` | capability needs an extra that isn't installed | no |
| `resource` | `MemoryError` | no |
| `internal` | anything unclassified — a bug | yes |

## Classification
One exception-type table for every capability. The handshake raises `host_incompatible`
directly; it is never classified.

| Exception | `kind` |
|---|---|
| `ValueError`, `TypeError`, `LinAlgError` | `invalid_input` |
| `KeyError`, `PatsyError` | `invalid_param` |
| `MemoryError` | `resource` |
| `ImportError` | `unsupported` |
| anything else | `internal` |

## Rules
1. Callers map an unknown `kind` to `internal`, so a newer host can add kinds without
   breaking an older caller.
2. Only `internal` carries a traceback. Every other kind is actionable without one.
3. No `cancelled`: the caller kills the host, so the host cannot report it.

## Warnings
Every call runs under `warnings.catch_warnings(record=True)`. Successful responses carry
`warnings: [{category, message, count}]`, deduplicated and capped. The caller decides where
they go.
