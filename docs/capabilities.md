# Capabilities

A capability is one curated, typed, versioned scikit-bio operation. There is no `eval` and no
generic call path (DESIGN §1).

## Contract
- Typed params and an Arrow table in, an Arrow table out.
- Pure: the same request returns the same answer.
- Knows nothing of transport, JSON, or error mapping ([`architecture.md`](architecture.md)).
  Raise ordinary exceptions; [`errors.md`](errors.md) classifies them.
- Declares a `schema_version`, bumped on any output schema change.

## Seeds
Why: DESIGN §3.7.

Every stochastic capability takes an integer `seed`. Omitted means a fixed default, never
nondeterminism. No `Generator` crosses the boundary.

## Result shapes
Why: DESIGN §3.10.

Where scikit-bio returns several related frames, return one long table with a discriminator
column. Differential abundance:

`feature_id, test, term, lfc, se, w, pvalue, qvalue, signif`

- `test` ∈ {`main`, `global`, `dunnett`, `pairwise`, `trend`}
- `term`: covariate or comparison; NULL where not applicable
- `lfc`, `se`: NULL for `global` and `trend`
- `w` is a z-statistic under `main`, chi-square/F under `global`; filter by `test`

A per-family rule, not a global one: other capabilities may return a single narrow table.

## Registry
`CAPABILITIES` in `src/skb_arrow/registry.py` maps name → `schema_version`. Empty in M1. M2
adds `echo`, which exists only to exercise the machinery; M3 adds `ancombc` and defines the
registration template and param spec.

## Open (M3)
- `schema_version` covers only the output schema. Changing a param default — including the
  default seed — changes answers without a version bump. Decide whether param specs are
  versioned.
