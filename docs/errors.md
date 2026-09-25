# Errors and warnings

Why one general mechanism: DESIGN §3.5–3.6. Implementation: `src/skb_arrow/errors.py`.

Every failure returns one structured envelope carrying a `kind`. Never per-capability.

| `kind` | Cause |
|---|---|
| `host_incompatible` | protocol violation; protocol or capability drift |
| `invalid_input` | data fails validation |
| `invalid_param` | bad formula, unknown method or param, out-of-range value |
| `unsupported` | capability needs an extra that isn't installed |
| `resource` | memory or disk exhausted |
| `internal` | anything unclassified — a bug |

## Classification
Host machinery (parsing, transport, dispatch):
1. skb_arrow error classes (`HostIncompatible`, `InvalidInput`, `InvalidParam`,
   `Unsupported`) → their kind.
2. Resource exhaustion → `resource`.
3. Anything else → `internal`.

Exceptions escaping a capability's `run`:
1. skb_arrow error classes → their kind.
2. Resource exhaustion → `resource`.
3. Raised in a `skb_arrow` module (the innermost traceback frame's `__name__`) → `internal`.
4. Raised in third-party code, by type (first match along the MRO):

| Exception | `kind` |
|---|---|
| `ValueError` | `invalid_input` |
| `KeyError`, `patsy.PatsyError` | `invalid_param` |
| `ImportError` | `unsupported` |
| anything else | `internal` |

Resource exhaustion is `MemoryError`, or `OSError` with errno `ENOSPC`, `EDQUOT`, or `ENOMEM`.
`numpy.linalg.LinAlgError` and pyarrow's `ArrowInvalid` are `ValueError`s. `PatsyError` is
matched by qualified name, so the host never imports patsy. `TypeError` is `internal`: it
signals a programming error, and signature drift raises it inside library wrapper frames.

A bad method or out-of-range value is `invalid_param` only when registry param validation
catches it. Caught inside scikit-bio instead, it arrives as `ValueError` → `invalid_input`.

**Capability contract:** raise `InvalidInput`, `InvalidParam`, or `Unsupported` for faults you
anticipate, including a missing extra (`except ImportError: raise Unsupported(...)`). Anything
else raised in capability code is a bug and reports as `internal`. The rules lean toward
reporting a caller error as a bug.

Residual: host misuse of a library that the library reports as `ValueError`, `KeyError`, or
`ImportError` from its own frames (a wrong column name passed to pandas) maps by type, as a
caller error. `ExceptionGroup` is `internal` whatever it wraps.

Only `Exception` is caught; `KeyboardInterrupt` and `SystemExit` end the host.

## Rules
1. Callers map an unknown `kind` to `internal`, so a newer host can add kinds without
   breaking an older caller.
2. Only `internal` may carry a traceback; callers must not assume one. Every other kind is
   actionable without one.
3. No `cancelled`: the caller kills the host, so the host cannot report it.
4. `message` is the error text for skb_arrow classes, `"Type: text"` otherwise. Text whose
   `str()` fails becomes `<Type: str() failed>`.

## Warnings
- Recorded across a whole call: reading inputs, running, writing outputs.
- The filters in effect keep their order, so the first match still wins: `ignore` stays
  `ignore`; every other action becomes `always`. So repeats are counted, `error` filters never
  raise, and Python's defaults keep `DeprecationWarning` (a host concern) hidden.
- Entry: `{category, message, count}`. `category` is the class name for builtins
  (`UserWarning`), else `module.QualName`.
- Deduplicated by (category, message) in first-seen order. After 20 distinct entries, one
  `{"category": "skb_arrow.WarningsTruncated", "message": "warnings beyond the first 20
  distinct", "count": <occurrences>}`. Memory is bounded by the 20 entries, not the flood.
- Carried on results and errors.
- Filters a library installs during a call are dropped when it ends, so capabilities import
  their libraries at module level and the host imports every capability before `ready`.
- Context-aware warnings (`-X context_aware_warnings`, default on free-threaded builds) are
  unsupported and fail loud.
