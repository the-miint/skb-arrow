# skb-arrow — Design

**Status:** M1 (foundation), M2 (protocol and transport), and M3 (first capability: `ancombc`)
complete; M4 next.
**Scope of this document:** skb-arrow only. The duckdb-miint integration (C++ submit
framework, `install_skb_arrow()`, SQL wrappers) is deliberately out of scope and gets its
own plan in that repository. **The protocol specification in this repo is the contract
between them.**

---

## 1. What skb-arrow is

skb-arrow is a **process host that exposes scikit-bio to non-Python callers over Arrow.**

A caller starts `skb-arrow` as a child process, speaks a small JSON line protocol on
stdin/stdout, and exchanges data as Arrow IPC through memory-mapped files. Requests name a
*capability* (`ancombc`, `mantel`, …); the host runs the corresponding scikit-bio code and
returns Arrow.

It exists because scikit-bio is Python and its callers may not be. Embedding CPython in the
caller was evaluated and rejected — see §3.1.

### Non-goals

- **Not a general Python RPC bridge.** Capabilities are curated, typed, and versioned. There
  is no `eval`, and no generic "call any function" escape hatch.
- **Not miint-specific.** miint is the first consumer, not the only intended one. Nothing in
  this repo may reference miint internals.
- **Not stateful.** Every call is a pure function: Arrow in, Arrow out. No sessions, no
  handles, no cached objects. See §3.3.
- **Not WASM-capable, ever.** The design requires `fork`/`exec`. This is an architectural
  incompatibility, not a porting gap. Do not contort the design for it.
- **Not a GPU scheduler.** GPU/array-API backends may become an install-time extra; they are
  not part of the protocol.

---

## 2. Architecture

```
┌──────────────────────────┐
│ caller (e.g. duckdb-miint)│
│                          │
│  control: JSON lines ────┼──── stdin/stdout ────┐
│  data:    Arrow IPC  ────┼─┐                    │
└──────────────────────────┘ │                    │
                             │                    ▼
              mmap'd files   │        ┌───────────────────────────┐
              (chunked)      └───────▶│ skb-arrow                 │
                                      │  ├ protocol/  envelopes   │
                                      │  ├ transport/ mmap + IPC  │
                                      │  ├ registry/  capabilities│
                                      │  └ capabilities/          │
                                      │       ancombc, mantel, …  │
                                      └───────────────────────────┘
                                                  │
                                          scikit-bio + numpy/scipy
```

Four internal layers, each independently testable; table and layering rule in
[`architecture.md`](architecture.md). The layering is the main maintainability lever: a
capability author writes one function and touches no transport, no JSON, no error mapping.

**Revised in M2:** a capability takes named tables, not one. `ancombc` needs sample metadata
beside its feature table, and `mantel` compares two distance matrices; folding either into one
table would push reshaping onto every caller. Protocol v1 carries named inputs from the start,
by §3.4's reasoning: no breaking bump later.

---

## 3. Decisions, with evidence

This section is the decision log. It exists so the reasoning survives context loss. Each
entry records what was chosen, why, and what measurement supports it.

### 3.1 Process host, not embedded CPython

Both nanobind and pybind11 were prototyped end-to-end driving `skbio.stats.composition.ancom`.

- nanobind **does not support embedding** (no `embed.h`; upstream docs: *"The ability to embed
  Python in an executable … is unsupported"*). It works via manual `nb::detail::init`, but as
  an unsupported configuration.
- pybind11 supports it properly (`py::scoped_interpreter`), and was the better of the two:
  200 K binary vs 216 K, 116 undefined Py symbols vs 174, no numpy headers needed at build time.
- Neither parallelises. Single-GIL threading measured **111 / 227 / 452 / 923 ms** for
  1/2/4/8 threads on fixed per-thread work — fully serialised.
- pybind11 subinterpreters (PEP 684) *do* parallelise — **124 / 124 / 127 / 147 ms** for
  1/2/4/8 — but **numpy cannot be imported into an own-GIL subinterpreter**: clean
  `ImportError`, then teardown corrupts the heap and aborts (reproduced deterministically on
  3.12 and 3.13). Unusable for a numpy-based stack.

Decisive factor was **blast radius**, not speed: `ancom` at 200×2000 took 18–22 s at **13 GB
peak RSS**. In-process that memory is invisible to the caller's memory accounting and an OOM
kills the caller. Out-of-process it kills a child.

The boundary is free. Measured: JSON RPC round trip **0.01 ms**, segment create+write+unlink
**0.02 ms**, and a full `ancom` batch through the host **52.9 ms** against **58.8–77.3 ms
in-process**. The process boundary cost nothing measurable.

### 3.2 Transport: memory-mapped files, not POSIX shm

gpl-boundary uses POSIX shm. That choice does not transfer, for a reason found by measurement
on macOS arm64 (Darwin 24.6, M-series):

- **POSIX shm cannot be handed off by name on macOS.** Creating a segment works (4096 MB
  succeeded), but closing the creator's fd and reopening by name fails with
  `ENXIO / Device not configured` — at every size. That create-then-reopen pattern *is* the
  handoff pattern a protocol needs.
- gpl-boundary avoids this only because it passes an **inherited fd** through `fork`
  (`shm.hpp`: "does NOT call `fstat` on the inherited fd"). So the real choice is
  fd-inheritance vs name-based handoff, and name-based is simpler.
- Confirmed macOS shm constraints: **31-char** name limit (Linux allowed 200+), and a second
  `ftruncate` is **refused** (`EINVAL`).
- On Linux the question is moot: `shm_open("/x")` literally creates a regular file at
  `/dev/shm/x`. Verified. Same mechanism, different API.

Memory-mapped files instead:

- Zero-copy and fast: `pa.memory_map` + `open_stream` + `read_all` on a 208 MB payload =
  **0.17 ms**, versus **26.45 ms** copying.
- **No doubling.** A 2 GiB payload consumed **2062 MB total** on Linux; 1024 MB consumed
  900 MB on macOS. RSS *looks* doubled because it counts the same physical pages the page
  cache already holds — `MemAvailable` is the honest number.
- **Cleanup of opened segments is crash-proof with no registry.** Both sides open, the
  consumer `unlink`s immediately, the kernel reclaims on last close. Verified
  valid-after-unlink on ext4, tmpfs, and APFS. This removes gpl-boundary's signal-safe
  cleanup registry and PID sweeping entirely — which matters because Python signal handlers
  are not async-signal-safe.

**Known gap** (M1 review): a segment its consumer never opens is never reclaimed — the host
killed after writing a response, or a request rejected before its segments are opened. M2
closes it with a session directory and receiver-disposes
([`transport.md`](transport.md#session-directory)); what remains is listed under
[residuals](transport.md#residuals).

Placement rules: [`transport.md`](transport.md). Linux uses tmpfs: RAM-backed, ordinary file
semantics.

**Known cost:** on macOS a 1024 MB cold read took **1689.6 ms** versus 63.5 ms at 512 MB — a
26× nonlinear jump once the payload exceeded available cache (8.4 GB free of 36 GB). Warm
re-read was 38.6 ms. This is why payloads are chunked (§3.4). macOS `mmap` also costs
6–10 ms versus 0.04 ms on Linux; negligible.

### 3.3 Stateless: pure functions only

No handles, no retained objects, no sessions beyond process lifetime. Simplest protocol,
trivially poolable, and a crash loses one call.

Accepted consequence: chained work re-materialises intermediates through the caller. A
distance matrix is O(n²) and may be the largest object in play. Revisit only with measurement.

### 3.4 Chunked payloads from the start

Payloads are lists of size-capped segments ([`transport.md`](transport.md)) from protocol
version 1. Arrow IPC streams are already multi-batch, so this is natural rather than
bolted on, and it avoids the §3.2 macOS cliff by construction instead of requiring a breaking
protocol bump later.

### 3.5 Errors are one general mechanism

Never per-capability: the caller maps `kind` to its own error type once. A single
exception-type table suffices because scikit-bio validates pervasively through `ValueError`.
Two durability rules keep it stable across versions — unknown kinds degrade, and only bugs
carry tracebacks — and `cancelled` is absent because the host is killed, not asked (§3.8).
Kinds, classifier, and rules: [`errors.md`](errors.md).

**Revised in M2** (M1 and M2 review evidence): host bugs raise `ValueError`, `TypeError`, and
`KeyError` too, so a type table alone reports them as caller errors without a traceback.
Classification is now scoped — machinery errors are `internal` unless explicit — and, for
capability code, keyed on the raising frame: raises from skb_arrow frames are bugs, and only
third-party raises map by type. `TypeError` became `internal` because signature drift
through a library decorator raises it in the library's frame (reproduced).

### 3.6 Warnings are collected, always

Warnings are returned, not printed, so the caller decides where they go (miint routes them to
`miint_warnings()`). They are deduplicated and capped because pandas `FutureWarning`s and numba
performance warnings can flood. Spec: [`errors.md`](errors.md#warnings).

### 3.7 Seeds: reproducible by default

scikit-bio's uniform `seed=` convention (via `get_rng()`) makes this cheap. Reproducible by
default because a declarative caller expects the same request to return the same answer.

Documented divergence: statelessness means only an integer crosses the boundary, so results
will not match a Python script that threads one RNG through several calls. For a declarative
caller this is the better semantic. Rules: [`capabilities.md`](capabilities.md#seeds).

### 3.8 Cancellation: the caller kills the process

No cooperative cancel. A long call sits inside one opaque numpy/scipy frame that would never
check a flag, so cooperative cancellation would offer weak guarantees for real complexity.
Mitigation: a persistent `NUMBA_CACHE_DIR` means a kill costs imports (~0.8 s), not JIT
recompilation.

**Revised in M4** (M4 design review, scikit-bio 0.7.4): the mitigation never worked. No
scikit-bio `@njit` sets `cache=True`, so `NUMBA_CACHE_DIR` stores nothing, and a restarted
host compiles each kernel again on its first call, about 0.3 s each (0.26–0.5 s measured).
Accepted: a kill costs imports plus that. `cache=True` upstream is the fix, outside this repo.

### 3.9 Python 3.14 only, uv-managed, PyPI-distributed

`requires-python = ">=3.14"`. Verified that **every** dependency ships cp314 wheels for both
linux-x86_64 and macos-arm64: scikit-bio, numpy, scipy, pandas, pyarrow, numba, llvmlite,
statsmodels (patsy is pure-python). Nothing builds from source.

3.14 also makes `SharedMemory(track=False)` and friends unconditional, removing the
`resource_tracker.unregister` hack that 3.12 requires.

Distribution is PyPI with trusted publishing; installation is `uv tool install skb-arrow`.
uv provides the interpreter, so the host never depends on a system Python.

Dependency tiers: base is skbio+numpy+scipy+pandas+pyarrow (~124 MB) **plus numba**
(+64 MB, always installed). jax/torch are deferred — as of scikit-bio 0.7.3 the only
array-API/GPU-capable functions are the eight composition transforms, and the caller
implements those natively. Because tiering is an install-time concern, adding them later
changes no architecture.

**Revised in M3** (measured with scikit-bio 0.7.4, Linux x86_64): something does build from
source. biom-format, a hard scikit-bio dependency, has no cp314 wheels (2.1.17, the latest),
so every install compiles it and needs a C compiler; CI has one. Wheels upstream or a compiler
policy is due before M5. The runtime install is 710 MB, not ~124 + 64 MB: numba is 189 MB of
it, llvmlite alone 172 MB. numba is installed and pinned now, though no capability uses it
before M4; scikit-bio imports it when present. Importing the registry takes 0.45 s warm,
numba's 0.08 s included, and over a second cold, paid by every host start and `--version`.
The libraries that compute answers are pinned exactly
([`capabilities.md`](capabilities.md#versioning)).

**M4:** `mantel` is numba's consumer, fixed to `engine="numba"`: on arm64, clang fuses
multiply-adds in the cython kernel, so the answer could depend on the engine. In the M4 design
review, numba's results were identical across square and condensed layouts and 1 and 32
threads (2000 IDs). CI runs it end to end on both platforms.

### 3.10 Result shapes: unify a family into one long table

One table with a discriminator column, rather than multiple outputs, where scikit-bio returns
several related frames. For differential abundance this covers `ancombc`'s two frames today
and extends to `ancombc2`'s post-hoc tests without a new shape. The one wrinkle — `w` changes
meaning with `test` — is accepted: the R package has the same property. A per-family
judgement, not a global rule. Schema: [`capabilities.md`](capabilities.md#result-shapes).

### 3.11 Formulas: bare column names only

patsy evaluates a formula's factors, the left-hand side included, as Python in scikit-bio's
frame: `__import__('os').getpid()` in a formula ran inside `ancombc` (reproduced, scikit-bio
0.7.4, patsy 1.0.3). That is the `eval` §1 rules out. `ModelDesc.from_formula` parses without
evaluating anything, left-hand-side payloads included (verified), so a formula is checked
there first.

Name lookup puts the data first: a column named `np`, `table`, `len`, `C`, or `center` wins
over scikit-bio's locals, builtins, and patsy's helpers (verified). What slips past a bare
name: Python NFKC-normalizes identifiers, so a column `ｔａｂｌｅ` resolved to scikit-bio's local
`table` and silently fit the feature matrix as 21 covariates (reproduced); `__debug__` and
keywords are constants; and `table` with no such column also fits that local. Hence a
factor must be a normalized identifier naming a metadata column. The cost is transforms and
`C()` contrasts, which a caller computes in SQL, setting a reference level by dictionary
order. Rules: [`capabilities.md`](capabilities.md#formulas).

### 3.12 Feature tables: long, densified by the host, sorted, Fortran order

A caller holds a sparse table long, `(sample_id, feature_id, value)`, as SQL does. Densifying
in the caller, only to have it reshape the answer back, is work every caller would repeat, so
the host densifies: an absent cell is 0, and `pseudocount` is added to every cell.

Exact parity with a direct scikit-bio call depends on layout. On the Atlas data, a C-ordered
float64 matrix differed from the natural DataFrame call by up to 3e-9; a Fortran-ordered one,
a single-block DataFrame's layout, matched bit for bit, as int64 or float64. Sample and feature
order moved results by about 4e-10, so both are sorted and SQL row order cannot change an
answer. Contract: [`capabilities.md`](capabilities.md#input-tables).

M3 asked for "no all-zero rows/columns"; the strictly positive contract implies it. A
non-negative variant waits for its first consumer, `dirmult_ttest` (M6).

### 3.13 stderr: relayed by a drainer process, never blocking the host

gpl-boundary deadlocked on an undrained stderr pipe, and so did this host (reproduced in M4,
Linux): a caller that never read its stderr pipe got no answer once the host wrote 4 MiB, by
`print`, `os.write(1)`, a libc `write(2)` holding the GIL, or a child process. A drain needs
somewhere to run that the writer can't stop:
- A thread in the host can't be it: C code blocked on fd 2 while holding the GIL keeps that
  thread from ever running.
- Writing only when `poll` reports room wedged on an unread pty, which reported room it
  didn't have, and races when hosts share one pipe: both see room, the second blocks.

So the host forks a drainer at startup, before any import beyond the standard library: no
thread exists yet, and GNU OpenMP kills a bare-`fork()` child once it has started. The drainer
reads a relay that replaces the host's fds 1 and 2, holds at most 1 MiB, drops the oldest
beyond, and writes with a blocking thread of its own. In the design review, all four floods
were answered in 0.01 s, kept plus dropped bytes equalled bytes written, the last words
survived, and with SIGINT ignored the host's `KeyboardInterrupt` traceback reached stderr in
20 of 20 runs (0 of 20 without). Rules: [`protocol.md`](protocol.md#channel).

**Revised in the M4 Phase 1 review.** Cutting at the last newline of what had arrived wrote a
line that reached the relay in two pieces (`abc`, then `def\n`) as two writes, and under full
CPU load the whole-lines test failed 7 of 30 runs; a partial line now waits up to 50 ms for its
newline (0 of 30). A flat 1 s after EOF cut off a stderr read at 500 KiB/s, last words lost and
no note; the drainer now gives up only after 1 s without a write. It is forked twice, so host
code reaping every child never waits on it, and closes every fd it inherits but its own, so a
channel copy leaked to the host can't hold the caller's EOF.

### 3.14 Lifecycle: the response pipe is watched, fork children are cut off

A caller killed mid-call left the host computing until the call ended, only then to fail
writing its answer. The host learns of it from its response pipe, not from its parent's
death: the caller need not be the parent, and a caller that stops reading but lives counts
too. Measured in the M4 design review (Linux, Python 3.14):
- `poll` with no events asked reports `POLLERR` on a pipe whose reader closed and `POLLHUP` on
  a socketpair whose peer closed; a full pipe with a live reader reports nothing, so an idle
  watch costs nothing. XNU's source suggests macOS `poll` reports neither without asked-for
  events (unmeasured), so there the host uses kqueue `EVFILT_WRITE` with `EV_CLEAR`, waking on
  `EV_EOF`.
- Acting on it at any time made every normal shutdown exit 1 (30 of 30 runs: the caller
  closes stdout while the host is still reading stdin's EOF). So the watch acts only while a
  capability runs: the request loop holds `Session.lock` except around `run`, and the watcher
  takes it and never gives it back (0 of 30). It also can't clean DIR mid-write.
- A capability blocked in one GIL-holding frame is abandoned only when the frame returns;
  scikit-bio's `mantel` kernels let the watch run.
- The watch is a thread, so the host is multi-threaded: `os.fork()` in a capability raises
  CPython's fork-with-threads `DeprecationWarning`, hidden unless warnings are on (`-X dev`).
  numba's and OpenMP's worker threads bring the same. The watch holds no lock a child could
  inherit, except while it cleans DIR on its way out.

A bare `fork()` child of a capability held the caller's EOF until it exited (3 s in the
review's probe). An at-fork hook now points the child's channel fds at `/dev/null` (0 s);
closing them instead would leave the Python objects owning numbers the child may reuse.
`multiprocessing` execs its children on Linux 3.14 (forkserver) and macOS (spawn) anyway.
Rules: [`protocol.md`](protocol.md#channel).

### 3.15 Segment sizing: bytes as written

M2 capped a segment's sum of batch `nbytes`, which counts a shared dictionary in full for every
batch: batches sharing one near the cap's size packed one per segment, each rewriting it (100
segments, 552,800 bytes, against 23,744 as one). The stream writer writes an unchanged
dictionary once per stream, so in the M4 design review counting a dictionary once per segment
still wrote 100 segments, while counting bytes as written, a segment's schema and first
batch's dictionaries exempt, wrote 7. The host measures with pyarrow's own serializer
(`MockOutputStream`, `get_record_batch_size`), which keeps no bytes; it rebases a slice's
offsets and bitmaps, as a real write does (42 MiB peak for a 10M-row string slice).

M2's 75% rule stays, on messages. A slice's message doesn't carry its dictionary, but it
carries a view column's data buffers whole, and a wide schema's per-message metadata
doesn't halve. Without the rule (the Phase 3 review), 800 `string_view` rows (95 KB) at a
64 KiB cap wrote 800 segments, 66 MB, and 300 `int8` columns × 1000 rows at 8 KiB wrote 1000
segments, 32 MB; with it, 1 segment, 95 KB, and 32, 1.3 MB.

No protocol bump: `segment_bytes` was always a target, not a bound, and the wire format is
unchanged.

The default cap was settled by timing, on a rule fixed before the run: the largest cap ≤ 512
MiB whose macOS 2 GiB read time is within 1.1× of the fastest, keeping 256 MiB if it
qualifies; and if whole-table reads were flat across caps but streamed reads weren't, §3.4
would be revised to say chunking serves streaming callers. The `large` workflow (manual
dispatch) runs the multi-GiB tests and `tests/segment_timing.py` on both platforms.

Its first run (37062727264, `f86f4aa`) kept 256 MiB. 2 GiB, medians of 3, seconds; macOS
(5 CPUs, 14 GiB) read cold from APFS after `purge`, Linux (2 vCPUs, 7.8 GiB) from tmpfs:

| Cap | Segments | macOS whole | macOS streamed | Linux whole | Linux streamed |
|---|---|---|---|---|---|
| 16 MiB | 171 | 0.630 | 0.696 | 0.184 | 0.262 |
| 64 MiB | 35 | 0.907 | 0.655 | 0.175 | 0.251 |
| 256 MiB | 9 | 0.686 | 0.664 | 0.174 | 0.247 |
| 1024 MiB | 3 | 1.253 | 0.682 | 0.175 | 0.246 |
| one segment | 1 | 0.682 | 0.732 | 0.186 | 0.189 |

256 MiB qualified in both macOS modes, whole by a thin margin (0.686 against 0.693). macOS
cold reads are noisy, varying 2× with no trend in the cap: weak evidence, but no cap stood
out. §3.4 stands. On macOS the streamed reads were the flat ones. On Linux streamed reads
were slower chunked, but that is each segment unmapped as the read moves on, which one
segment pays after the timer stops: holding every batch until then, a local 2 GiB streamed
read took 0.097–0.099 s at 171, 9 and 1 segments (0.112–0.115 s chunked, released as read).

---

## 4. Versioning and compatibility

Adopted from gpl-boundary, which got this right: an integer `protocol_version` checked at the
init handshake, so drift fails immediately rather than as garbled batches later; a
per-capability `schema_version`; minimum-version gating; and forward-compatible handling of
unknowns in both directions. Rules: [`protocol.md`](protocol.md#versioning).

**Revised in M3** (M3 Phase 1 review): `schema_version` covers a capability's whole interface
— inputs, params with their defaults and accepted values, output schema — and every change
bumps it, additions included. A changed default changes answers as surely as a changed column,
the default seed most sharply. And minimum-version gating only works if a caller can learn
that a param it sends exists: under "additions need no bump", a caller sending a new param to
an older host with the same version gets `invalid_param`. The implementation is
`host_version`'s, with the libraries that compute answers pinned exactly. Rules:
[`capabilities.md`](capabilities.md#versioning).

---

## 5. Milestones

Every milestone obeys §6. Milestones are carved into separate plan documents as they start.

### M1 — Foundation
Repository, tooling, CI, and the documents that make later milestones cheap.

- `pyproject.toml` (hatchling, `requires-python = ">=3.14"`), `src/skb_arrow` layout, uv lockfile
- `CLAUDE.md` derived from duckdb-miint's — rules, principles, and §6 as the authoritative copy
- `docs/`: `architecture.md`, `protocol.md`, `transport.md`, `capabilities.md`, `errors.md`,
  referenced from `CLAUDE.md` so they load on demand instead of occupying context
- GitHub Actions: lint, format, type check, pytest — matrix **{linux-x86_64, macos-arm64} × 3.14**
- Release workflow for PyPI trusted publishing; publishes only on a non-prerelease GitHub
  release, and none is cut before M5
- `skb-arrow --version` reporting protocol version and an empty capability registry

**Done when:** CI green on both platforms; `uv tool install .` yields a runnable
`skb-arrow --version`; every doc above exists and is referenced from `CLAUDE.md`.

### M2 — Protocol and transport
The contract, with no scikit-bio in sight.

- Request/response envelopes; init handshake with `protocol_version` + capability registry
- Chunked Arrow IPC over mmap'd segments; unlink-after-open; platform placement policy
- Error taxonomy + classifier (§3.5); warning collection (§3.6)
- An `echo` capability that exists purely to exercise the machinery
- Tests: round-trip across chunk boundaries, every error `kind`, orphan-free teardown after
  SIGKILL, both platforms in CI

**Done when:** `echo` round-trips multi-segment Arrow on both platforms; every `kind` has a
test that produces it; no temp files survive a killed host.

### M3 — First capability: `ancombc`
Establishes the capability template every later one follows.

- Declarative registration: name, schema version, param spec, input contract
- `ancombc` with the §3.10 unified schema
- Parity tests against direct scikit-bio calls; seed determinism; error mapping for bad
  formulas and non-positive input
- Generic pre-flight input validation (strictly positive, no all-zero rows/columns) reporting
  offending IDs — general because it validates the table contract, not the capability

**Done when:** `ancombc` matches scikit-bio exactly on committed fixtures; its schema is
documented; adding a capability requires touching only `capabilities/` and the registry.

### M4 — Robustness and lifecycle
- Daemon lifecycle: init → many batches → shutdown; warm-process reuse
- stderr discipline (an undrained stderr pipe deadlocked gpl-boundary; do not repeat it)
- Chunk size policy, and large-payload tests sized against the §3.2 macOS cliff
- `mantel` as the numba consumer (no `NUMBA_CACHE_DIR`: §3.8)
- Soak test: many batches, no leaked descriptors or files

**Done when:** a soak run leaves no orphans; numba tier is exercised in CI; stderr cannot wedge
the host.

### M5 — Release
- PyPI publish via trusted publishing; version/compat policy (§4) enforced in code and tested
- `skb-arrow doctor` reporting interpreter, versions, extras, and transport placement
- Installation and capability documentation

**Done when:** `uv tool install skb-arrow` from PyPI works on both platforms from a clean machine.

### M6 — Capability expansion
`ancom`, `dirmult_ttest`, `ilr`/`ilr_inv`, `permdisp`, then `ancombc2` against a pinned
scikit-bio. Each follows the M3 template.

Note on `ancombc2`: scikit-bio 0.7.4 released it (**revised in M3**; it was on an unmerged
branch with defects when this plan was written). Its post-hoc tests apply the mdFDR adjustment
(verified). `trend` arrives with it: its contrast matrices need structured params. Independent
skbio pinning is exactly why this repo is separate.

---

## 6. Working agreement

Moved to [`CLAUDE.md`](../CLAUDE.md), the authoritative copy.
