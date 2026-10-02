# Protocol

The contract between skb-arrow and its callers. Implementation: `src/skb_arrow/protocol.py`.

## Channel
- The caller runs `skb-arrow` as a child process.
- Control: one JSON object per line. Requests on stdin, responses on stdout.
- stdout carries protocol messages only. Before importing anything heavy, the host moves the
  channel to private fds that exec'd children don't inherit; fd 0 then reads `/dev/null`, and
  fd 1 and `sys.stdout` go with fd 2 (below). Library prints, C-level writes to fd 1,
  `input()`, and exec'd children cannot reach the channel. A bare `fork()` child does hold
  it: if one outlives the host, the caller sees no EOF (M4 lifecycle).
- stderr carries diagnostics only, and never blocks the host (DESIGN §3.13). At startup,
  before anything but the standard library loads, the host forks a drainer that relays fds 1
  and 2, its own, its libraries', and its children's, to stderr:
  - A caller may read stderr lazily or never. Unread, the drainer holds at most 1 MiB,
    dropping the oldest, and on resuming writes `skb-arrow: N bytes of stderr dropped`.
  - Writes go out whole lines at a time, up to `PIPE_BUF` bytes, so hosts sharing one stderr
    pipe don't split each other's lines. A longer line is cut; one without its newline waits
    up to 50 ms for it, so a prompt still shows.
  - stderr reaches EOF once the host and every child holding its fds 1 and 2 have exited, and
    the drainer has written what it holds. Once they have exited, it gives up on a stderr that
    takes no write for 1 s, so a caller may wait for the host, then read.
  - fd 2 is a pipe, not the caller's terminal: output that checks for one (colour, progress
    bars) turns off.
  - The drainer ignores SIGINT, SIGTERM, and SIGHUP, so a process-group signal leaves it to
    relay the host's last words. It is never the host's child, so host code reaping every
    child doesn't wait on it; init reaps it: a caller that runs as PID 1 must reap orphans
    (`docker run --init`, tini).
  - A drainer that fails writes `skb-arrow: stderr drainer failed: …`; the host's writes to
    fds 1 and 2 then fail.
  - If stderr is the channel's own pipe or socket (`2>&1`), it is discarded, not mixed into
    responses: such a caller gets no diagnostics, not even a startup error. On macOS a TCP
    socket has no inode, so `2>&1` onto one goes undetected and corrupts responses.
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
come in request order, so a caller may pipeline, reading responses as it goes (or bounding
requests in flight): the host blocks on a response nobody reads. Shutdown: the caller closes
stdin, having
opened every output it needs, since the host then cleans DIR
([`transport.md`](transport.md#session-directory)).

| `type` | | Fields |
|---|---|---|
| `init` | → | `protocol_version` int; `segment_bytes` int ≥ 1024, optional (default 256 MiB) |
| `ready` | ← | `protocol_version`; `host_version` str; `capabilities` {name: `schema_version`} |
| `call` | → | `id` str or int, optional; `capability` str; `params` object, optional; `input` {table: [segment names]} |
| `result` | ← | `output` [segment names] |
| `error` | ← | `kind`; `message`; `traceback` (`internal` only); `protocol_version` (until an `init` succeeds) |

- Every response also carries `id` (the request's, if a string or int; else null) and
  `warnings` ([`errors.md`](errors.md#warnings)).
- `init` must succeed before any `call`, and only once. A failed `init` may be retried, so a
  caller can fall back to an older version.
- An int is a JSON integer, never `true`; a null field is absent.
- `host_incompatible`, and the host keeps serving: a line that isn't a UTF-8 JSON object
  (`NaN` and `Infinity` are not JSON), an unknown `type`, a missing or mistyped field, `call`
  before `init`, a second `init`, another `protocol_version`, `segment_bytes` below 1024, an
  unknown capability, or a segment rule broken ([`transport.md`](transport.md#segments)).
- `invalid_param`: a param the capability doesn't declare (even if null), a required param
  missing, a mistyped param or one that breaks its rule, or input tables other than those it
  declares ([`capabilities.md`](capabilities.md#registry)).
- Every string under a request's `input` is a segment it names. All are unlinked, whatever the
  outcome, before the capability runs. `output` is one table.
