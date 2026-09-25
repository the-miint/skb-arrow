# Capabilities

A capability is one curated, typed, versioned scikit-bio operation. There is no `eval` and no
generic call path (DESIGN §1).

## Contract
- Typed params and named Arrow tables in, one Arrow table out.
- Pure: the same request returns the same answer.
- Knows nothing of transport or JSON ([`architecture.md`](architecture.md)).
- Raises `InvalidInput`, `InvalidParam`, or `Unsupported` for faults it anticipates; anything
  else it raises is a bug ([`errors.md`](errors.md#classification)).
- Imports its libraries at module level, so their warning filters persist across calls.
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
`CAPABILITIES` in `src/skb_arrow/registry.py` maps name → `Capability(schema_version, inputs,
params, run)`. A call must name exactly the declared input tables and only declared params;
`run(tables, params)` receives them by name.

`echo` (input `table`, no params) returns its input; it exists only to exercise the machinery.
M3 adds `ancombc`, typed param specs, and the registration template.

## Open (M3)
- `schema_version` covers only the output schema. Changing a param default — including the
  default seed — changes answers without a version bump. Decide whether param specs are
  versioned.
