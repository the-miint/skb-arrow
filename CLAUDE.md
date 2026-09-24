# CLAUDE.md

skb-arrow is a process host exposing scikit-bio to non-Python callers over Arrow.
[`docs/DESIGN.md`](docs/DESIGN.md) holds the design, decision log (§3), and milestones (§5).
Rules live in `docs/` (below); DESIGN holds the reasons and evidence. Decisions in §3 are
settled: reopen one only with new evidence, and record it there.

## Hard requirements
- Never `rm` without permission.
- Never `git add -A`, `git add .`, or `git commit -a`. Stage explicit paths.
- Never change a failing test's expected value without permission.

## Priorities
1. Red/green/refactor TDD
2. Verifiably correct code
3. Maintainable code (DRY, KISS)
4. Performance

## Working agreement
Authoritative. Binds every milestone, phase, and session, including one resuming with no
memory of prior work.

1. **Think before coding.** State assumptions and success criteria. Ask rather than guess.
   Push back when a simpler approach exists. Stop when confused.
2. **Red/green/refactor.** Write the failing test first; see it fail for the intended reason.
   Then pass, then refactor. Name the input class that breaks each assertion — a test that
   cannot fail when the logic changes is not a test.
3. **Simplicity first.** Minimum code that solves the problem. No speculative abstraction, no
   abstraction for single-use code.
4. **Tight code, succinct prose.** Comments say only what the code cannot — why, not what.
   Docs are terse and scannable. No narration, no restating signatures, no filler.
5. **Surgical changes.** Touch only what the task requires. Don't improve adjacent code.
6. **Read before writing.** Read the layer's contracts, exports, and callers first. If
   existing structure is puzzling, ask.
7. **Match conventions.** Conformance over taste. If a convention is harmful, say so; don't
   fork silently.
8. **Respect the layering** ([`docs/architecture.md`](docs/architecture.md)). A capability
   that reaches into transport or JSON is a design regression.
9. **Surface conflicts, don't average them.** Pick the more recent or better-tested pattern,
   say why, flag the other.
10. **Fail loud.** "Complete" is false if anything was skipped, commented out, or silently
    xfail'd. Surface uncertainty.
11. **Self-triggered review.** Run `/code-review` at each phase boundary and whenever a
    change outgrows its plan. Present findings and stop; fix nothing until acknowledged.
12. **Stop and summarize at every phase end:** what was done, what is verified, what
    remains, every deviation from the plan and why. Don't start the next phase in the same
    breath.

## Commands
```
.githooks/pre-commit                             # full gate, exactly as CI runs it
uv sync --locked
uv run --locked pytest
uv run --locked ruff check
uv run --locked ruff format --check
uv run --locked mypy
git config core.hooksPath .githooks              # run the gate on every commit
```
Always `--locked`: without it uv silently rewrites `uv.lock`.

## Docs
Read on demand.
- [`docs/architecture.md`](docs/architecture.md) — layers, module map
- [`docs/protocol.md`](docs/protocol.md) — control channel, versioning
- [`docs/transport.md`](docs/transport.md) — Arrow IPC over mmap'd segments
- [`docs/errors.md`](docs/errors.md) — error kinds, classification, warnings
- [`docs/capabilities.md`](docs/capabilities.md) — capability contract, seeds, result shapes
