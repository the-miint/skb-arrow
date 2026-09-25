# Capabilities

A capability is one curated, typed, versioned scikit-bio operation. There is no `eval` and no
generic call path (DESIGN §1).

## Contract
- Typed params and named Arrow tables in, one Arrow table out.
- Pure: the same request returns the same answer ([versioning](#versioning) says how exactly).
- Knows nothing of transport or JSON ([`architecture.md`](architecture.md)).
- Raises `InvalidInput`, `InvalidParam`, or `Unsupported` for faults it anticipates; anything
  else it raises is a bug ([`errors.md`](errors.md#classification)).
- Imports its libraries at module level, so their warning filters persist across calls.
- Declares a `schema_version` ([versioning](#versioning)).

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
params, run)`; `params` maps each name to `Param(type, default, valid, rule)`. Before any input
is read, a call is checked in this order:
1. The capability exists, else `host_incompatible`.
2. Every param name is declared, even one whose value is null.
3. The input tables are exactly those declared.
4. Null values are dropped: null is absent.
5. Every param without a default is present.
6. Each value has the declared JSON type exactly: `true` is never an integer. A number
   (`float`) also takes an integer, and must convert to a finite float (`1e400` does not).
7. Each value passes `valid`; `rule` says what it requires.

Failures from 2 on are `invalid_param`: `unknown echo params: x`, `echo takes inputs: table`,
`ancombc param 'formula' is required`, `… must be a number`, `… must be in (0, 1)`.
`run(tables, params)` receives the tables by name and every declared param, defaults filled.

`echo` (input `table`, no params) returns its input; it exists only to exercise the machinery.

## Versioning
`schema_version` covers a capability's interface: its input tables and their contracts; its
params' names, types, defaults, and accepted values; its output schema. Bump it on any change
that could reject a request or change its answer. A new optional param whose default keeps
today's behaviour, or a newly accepted value, needs none.

`host_version` covers the implementation. Dependencies are pinned exactly
([`pyproject.toml`](../pyproject.toml)), so an answer is reproducible per `host_version`: bit
for bit on the same machine and BLAS thread count, to floating-point tolerance across them.
