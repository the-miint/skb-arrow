# skb-arrow — Design

**Status:** design accepted, implementation not started.
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

Four internal layers, each independently testable:

| Layer | Responsibility | Must not know about |
|---|---|---|
| `transport` | mmap'd segments, chunking, Arrow IPC encode/decode, unlink-after-open | capabilities, scikit-bio |
| `protocol` | request/response envelopes, init handshake, versioning, error taxonomy | scikit-bio |
| `registry` | capability declaration, param validation, schema versions | transport details |
| `capabilities` | one module per scikit-bio capability | transport, JSON |

The layering rule is the main maintainability lever: **a capability author writes a function
that takes typed params plus an Arrow table and returns an Arrow table.** They touch no
transport, no JSON, no error mapping.

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
- **Cleanup is crash-proof with no registry.** Both sides open, the consumer `unlink`s
  immediately, the kernel reclaims on last close. Verified valid-after-unlink on ext4, tmpfs,
  and APFS. This removes gpl-boundary's signal-safe cleanup registry and PID sweeping
  entirely — which matters because Python signal handlers are not async-signal-safe.

Placement: `/dev/shm` on Linux (tmpfs, RAM-backed, ordinary file semantics), `TMPDIR` on
macOS. Overridable.

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

A request carries a **list** of mapped segments, each holding Arrow IPC record batches, sized
under a policy cap. Arrow IPC streams are already multi-batch, so this is natural rather than
bolted on, and it avoids the §3.2 macOS cliff by construction instead of requiring a breaking
protocol bump later.

### 3.5 Errors are one general mechanism

Never per-capability. The host always returns a structured envelope with a `kind`; the caller
maps `kind` to its own error type once.

| `kind` | Cause | Traceback? |
|---|---|---|
| `host_incompatible` | protocol or capability schema drift | no |
| `invalid_input` | scikit-bio data validation, `LinAlgError` | no |
| `invalid_param` | bad formula, unknown method, out-of-range value | no |
| `unsupported` | capability needs an extra that is not installed | no |
| `resource` | `MemoryError` | no |
| `internal` | anything unclassified — a bug | **yes** |

Classification is a single exception-type table, roughly: `ValueError`/`TypeError`/`LinAlgError`
→ `invalid_input`; `KeyError`/`PatsyError` → `invalid_param`; `MemoryError` → `resource`;
`ImportError` → `unsupported`; else `internal`. This works because scikit-bio validates
pervasively through `ValueError`.

Two durability rules:
1. **Unknown `kind` must map to `internal` on the caller side**, so a newer host can add kinds
   without breaking an older caller.
2. **Only `internal` carries a traceback.** Every other kind must be actionable without one.

`cancelled` is deliberately absent: the caller kills the host, so the host cannot report it.

### 3.6 Warnings are collected, always

Every capability call is wrapped in `warnings.catch_warnings(record=True)`. Successful
responses carry `warnings: [{category, message, count}]`, deduplicated and capped — pandas
`FutureWarning`s and numba performance warnings can flood. The caller decides where they go
(miint routes them to `miint_warnings()`).

### 3.7 Seeds: reproducible by default

scikit-bio has a uniform `seed=` convention via `get_rng()`. Every stochastic capability
exposes `seed`, and **omitting it yields a fixed default, not nondeterminism** — the same
request must return the same answer.

Documented divergence: statelessness means only an integer seed crosses the boundary, never a
shared `Generator`. Results therefore will not match a Python script that threads one RNG
through several calls. For a declarative caller this is the better semantic.

### 3.8 Cancellation: the caller kills the process

No cooperative cancel. A long call sits inside one opaque numpy/scipy frame that would never
check a flag, so cooperative cancellation would offer weak guarantees for real complexity.
Mitigation: a persistent `NUMBA_CACHE_DIR` means a kill costs imports (~0.8 s), not JIT
recompilation.

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

### 3.10 Result shapes: unify a family into one long table

Where scikit-bio returns several related frames, return **one table with a discriminator
column** rather than multiple outputs. For the differential-abundance family:

`feature_id, test, term, lfc, se, w, pvalue, qvalue, signif`

- `test` ∈ `{main, global, dunnett, pairwise, trend}`
- `term` carries Covariate or Comparison; NULL where not applicable
- `lfc`/`se` NULL for `global`/`trend`

This covers `ancombc`'s two frames today and extends to `ancombc2`'s post-hoc tests without a
new shape. Documented wrinkle: `w` is a z-statistic under `main` but chi-square/F under
`global`. Consumers filter by `test`; the R package has the same property.

This is a per-family judgement, not a global rule. Other capabilities may legitimately return
a single narrow table.

---

## 4. Versioning and compatibility

Adopted from gpl-boundary, which got this right:

- **`protocol_version`** — integer, bumped on any wire-format change. The init handshake
  rejects drift immediately rather than producing garbled batches later.
- **Per-capability `schema_version`** — advertised in the init reply's capability registry, so
  a caller can refuse to bind a capability whose output schema it does not understand.
- **Minimum-version gating** — a caller may require a floor and fail fast with an actionable
  message.
- Unknown fields in requests are ignored; unknown error `kind`s degrade to `internal`. Both
  directions stay forward-compatible.

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
- Release workflow skeleton for PyPI trusted publishing (not yet publishing)
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
- `mantel` as the numba consumer, with a persistent `NUMBA_CACHE_DIR`
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

Note on `ancombc2`: it is not in released scikit-bio. It lives on the unmerged `ancombc2_dev`
branch and has known defects (`sensitivity_analysis()` raises `TypeError`;
`_mdfdr_pairwise`/`_mdfdr_dunnett` compute `R` but never apply the mdFDR level adjustment their
docstrings describe). Independent skbio pinning is exactly why this repo is separate.

---

## 6. Working agreement

**These rules apply to every milestone and every phase. They persist across context clears and
across sessions. If you are resuming with no memory of prior work, these still bind.**

1. **Red/green/refactor TDD.** Write the failing test first and *see it fail for the intended
   reason*. Then make it pass. Then refactor. A test that cannot fail when the logic changes is
   not a test — name the input class that breaks each assertion.
2. **Self-triggered code review.** Run `/code-review` at each phase boundary and at any point
   the change grows beyond what was planned. Present findings and **stop** — do not fix
   anything until the findings are acknowledged.
3. **Stop and summarize at every phase completion.** State what was done, what is verified, what
   remains, and **every deviation from the plan** and why. Do not begin the next phase in the
   same breath.
4. **Fail loud.** "Complete" is false if anything was skipped, commented out, or silently
   xfail'd. Surface uncertainty rather than hiding it.
5. **Simplicity first.** Minimum code that solves the problem. No speculative abstraction, no
   abstraction for single-use code.
6. **Surgical changes.** Touch only what the task requires. Do not improve adjacent code.
7. **Read before writing.** Read the layer's existing contracts and callers first. If existing
   structure is puzzling, ask rather than guess.
8. **Respect the layering** in §2. A capability that reaches into transport or JSON is a
   design regression, not a shortcut.
9. **Never `rm` without permission.** Never `git add -A`, `git add .`, or `git commit -a` —
   stage explicit paths, always.
10. **Surface conflicts, don't average them.** If two patterns contradict, pick the more recent
    or better-tested one, say why, and flag the other.

Design decisions in §3 are settled. Reopen one only with new evidence, and record the evidence
here when you do.
