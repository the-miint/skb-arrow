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
- Leaves process state as found, so an answer can't depend on the calls before it: it seeds
  no global RNG and changes no scikit-bio config, pandas option, working directory,
  environment variable, or numpy error mode. The host enforces nothing; a test runs every
  registered capability and checks.

## Input tables
Contracts shared by capabilities (`src/skb_arrow/capabilities/_tables.py`), checked before any
scikit-bio code runs. A violation is `invalid_input`, counted and shown by example:
`table: 3 of 40 cells are not positive and finite, e.g. ('s1', 'f2'), ('s1', 'f7'), ('s4', 'f2')`.
At most 5 examples, in sorted order.

**IDs** (`sample_id`, `feature_id`, `id_a`, `id_b`): integers or strings (dictionary-encoded, `large_string`,
and `string_view` included), normalized to `int64` or `string`. Never null.

**Feature table**, long: columns exactly `sample_id`, `feature_id`, `value`, in any order.
- `value` is integers or floating point, never null or negative. Decimal is rejected: cast it
  to DOUBLE. Values become float64, so an integer above 2**53 rounds.
- At least one row, and each (`sample_id`, `feature_id`) pair at most once.
- Densified to a samples × features matrix: an absent pair is 0, then `pseudocount` is added
  to every cell. Every cell must then be finite and positive. (A variant allowing zeros arrives
  with its first consumer, M6.)
- Samples and features are sorted ascending, so an answer never depends on row order. The
  matrix is float64 in Fortran order, a pandas DataFrame's layout, so results match a direct
  scikit-bio call bit for bit.

**Sample metadata**, wide: a `sample_id` column holding the table's kind of ID, each sample at
most once, and covariate columns.
- Every sample in the table must appear, and other samples are dropped. Every row's
  `sample_id` is checked, and each column's type; covariate values only for the table's
  samples.
- Covariates are booleans, integers, floating point, strings, or dictionaries of strings.
  patsy treats booleans and strings as categorical with levels sorted, and a dictionary as
  categorical in dictionary order: the first value in use is the reference level (a DuckDB ENUM
  sets it). Unused values are dropped; every chunk carries the same dictionary, without repeats
  or nulls.
- No covariate is null, NaN, or infinite for a sample in the table.

**Distance table**, long: columns exactly `id_a`, `id_b`, `distance`, in any order.
- `id_a` and `id_b` hold the same kind of ID.
- `distance` is integers or floating point, never null, NaN, infinite, or negative. Decimal
  is rejected: cast it to DOUBLE. Values become float64, so an integer above 2**53 rounds.
- Each unordered pair of distinct IDs appears exactly once, in either orientation: no row pairs
  an ID with itself, and none of the n(n−1)/2 pairs is missing or repeated (`(b, a)` repeats
  `(a, b)`). At least one row.
- Checked in this order: columns, IDs, their kinds, `distance`'s type, rows, self pairs, nulls,
  repeats, missing pairs, values. Memory stays proportional to the rows until every check
  passes.
- IDs are sorted ascending and the matrix is condensed, so an answer never depends on row
  order or orientation.

**Distance tables** compared with each other hold the same IDs, of the same kind.

## Formulas
Why: DESIGN §3.11.

patsy runs a formula's terms as Python, so a formula is checked before patsy evaluates it:
- It parses, has no left-hand side (`y ~ a`), and names at least one column (an empty
  formula, `0`, and `1` name none).
- Every factor is a metadata column's bare name: a Python identifier, not a keyword or
  `__debug__`, that NFKC normalization leaves unchanged (Python reads `ｔａｂｌｅ` as `table`).
- Operators `+ - * : / **` and `0`/`1` are allowed; calls are not (`log(age)`, `C(bmi)`,
  `Q('a b')`). Transform and rename in the caller; set a reference level by dictionary order.

A failure is `invalid_param`. A column named `Intercept` collides with the one patsy adds and
fails in scikit-bio, as `invalid_input`: rename it.

## ancombc
scikit-bio 0.7.4's `ancombc`, with its post-hoc tests. `schema_version` 1.

Inputs: `table`, a feature table; `metadata`, its sample metadata ([input tables](#input-tables)).

| Param | Type | Default | Accepts |
|---|---|---|---|
| `formula` | string | required | a [formula](#formulas) |
| `grouping` | string | none | see below |
| `posthoc` | array of strings | `[]` | distinct: `global`, `pairwise`, `dunnett` |
| `pseudocount` | number | 0 | ≥ 0 |
| `max_iter` | integer | 100 | ≥ 1 |
| `tol` | number | 1e-5 | > 0 |
| `alpha` | number | 0.05 | in (0, 1) |
| `p_adjust` | string | `holm` | `holm`, `bonferroni`, `bh`, `by` |
| `bootstraps` | integer | 100 | ≥ 1; `dunnett` only |
| `seed` | integer | 0 | ≥ 0; `dunnett` only |

- `grouping` is the covariate that post-hoc tests compare, and `posthoc` needs it. It must be a
  metadata column of strings or a dictionary (booleans have two groups at most), a term of its
  own in `formula`, and `formula` must have an intercept; each is `invalid_param`. The intercept
  rule is conservative: `0 + region + bmi` still codes `bmi` against a reference, but is
  rejected; `region + bmi` is the same model. `grouping` must have at least three groups among
  the table's samples, or scikit-bio raises `invalid_input`.
- Checked in this order: the formula's shape, and that `posthoc` has a `grouping`; the tables;
  the formula's columns, and `grouping`.
- `p_adjust` offers scikit-bio's own methods only: the statsmodels ones fail on the fractional
  family sizes of the post-hoc tests' correction.
- The table contract admits no zero cell, so a table with zeros needs a `pseudocount`
  (the R package's tutorial uses 1).

Output, ordered by `test` (`main`, then `posthoc`'s in the order `global`, `pairwise`,
`dunnett`), then feature, then term:

| Field | Type | Null |
|---|---|---|
| `feature_id` | the table's, normalized | never |
| `test` | string | never |
| `term` | string | for `global` |
| `lfc`, `se` | double | for `global` |
| `w`, `pvalue`, `qvalue` | double | never |
| `signif` | boolean | never |

- `main`: a row per feature and design column (`Intercept`, `region[T.NE]`, `age`).
- `pairwise`: each `grouping` column against the reference level, then each pair of them
  (`bmi[T.obese]_bmi[T.overweight]`, obese minus overweight). `dunnett`: each against the
  reference.
- `lfc` is a natural-log fold change.
- NULL means not applicable. NaN and ±inf are scikit-bio's, passed through as IEEE values.
  Nothing flags a coefficient the data can't estimate: a collinear design yields finite
  pseudo-inverse numbers.
- `pairwise` and `dunnett` correct only across the features a screen keeps; a feature it drops
  has `pvalue` 1. Each has its own screen, not the `global` rows: `pairwise` a global test
  adjusted by `bh` whatever `p_adjust` is, `dunnett` a bootstrap test of each feature's largest
  |`w`|.
- `dunnett` is the only stochastic test.

## mantel
scikit-bio 0.7.4's `mantel`, on its numba kernels. `schema_version` 1.

Inputs: `x`, `y`, distance tables over the same IDs ([input tables](#input-tables)).

| Param | Type | Default | Accepts |
|---|---|---|---|
| `method` | string | `pearson` | `pearson`, `spearman`, `kendalltau` |
| `permutations` | integer | 999 | ≥ 0 |
| `alternative` | string | `two-sided` | `two-sided`, `greater`, `less` |
| `seed` | integer | 0 | ≥ 0 |

- `x` is permuted. `greater` tests for a positive correlation, `less` for a negative one.
- Checked in this order: `x`, `y`, then that they hold the same IDs. Fewer than 3 IDs is
  scikit-bio's `invalid_input`.
- `kendalltau` permutes in Python, not numba: about 52 s at 1000 IDs and 999 permutations.
- scikit-bio draws every permutation up front: (`permutations` + 1) × IDs × 8 bytes, 16 GB
  for a million at 2000 IDs. Unbounded, as in scikit-bio and R; one too large to size at
  all (`2**62`) reports `invalid_input`, not `invalid_param`.
- numba's threads come from the environment (`NUMBA_NUM_THREADS`, default every core); the
  answer doesn't depend on their count. A host's first call compiles each kernel it uses,
  about 0.3 s each (DESIGN §3.8).

Output, one row:

| Field | Type | Null |
|---|---|---|
| `statistic` | double | never |
| `pvalue` | double | never |
| `n` | int64 | never |

- `n` is the number of IDs.
- `pvalue` is NaN for `permutations: 0`. A constant input makes `statistic` and `pvalue` NaN,
  with scipy's `ConstantInputWarning` for `pearson` and `spearman`; `kendalltau` warns
  nothing.

## Seeds
Why: DESIGN §3.7.

Every stochastic capability takes an integer `seed`. Omitted means a fixed default, never
nondeterminism. No `Generator` crosses the boundary.

## Result shapes
Why: DESIGN §3.10.

Where scikit-bio returns several related frames, return one long table with a discriminator
column. Differential abundance:

`feature_id, test, term, lfc, se, w, pvalue, qvalue, signif`

- `test` ∈ {`main`, `global`, `pairwise`, `dunnett`}, in that order; `trend` arrives with
  `ancombc2` (M6)
- `term`: covariate or comparison; NULL where not applicable
- `lfc`, `se`: NULL for `global` (and `trend`)
- `w` is a z-statistic under `main`, chi-square/F under `global`; filter by `test`

A per-family rule, not a global one: other capabilities may return a single narrow table.

## Registry
`CAPABILITIES` in `src/skb_arrow/registry.py` maps name → `Capability(schema_version, inputs,
params, run)`; `params` maps each name to `Param(kind, default, valid, rule, item)`. Before any
input is read, a call is checked in this order:
1. The capability exists, else `host_incompatible`.
2. Every param name is declared, even one whose value is null.
3. The input tables are exactly those declared.
4. Then each param in turn, in declaration order:
   1. Present, unless it has a default. Null is absent. A null default makes a param
      optional with no value: `run` gets `None`.
   2. Of the declared JSON type exactly: `true` is never an integer. An integer fits 64 bits.
      A number (`float`) also takes an integer, and must convert to a finite float (`1e400`
      does not). An array (`list`) holds items of its `item` type.
   3. Passes `valid`; `rule` says what it requires.

Failures from 2 on are `invalid_param`: `unknown echo params: x`, `echo takes inputs: table`,
`ancombc param 'formula' is required`, `… must be a number`, `… item 1 must be a string`,
`… must be in (0, 1)`. `run(tables, params)` receives the tables by name and every declared
param, defaults filled; each call gets its own copy of a default.

A declaration is checked on import: `kind` is a JSON scalar type, or `list` with a scalar
`item`; `valid` comes with a `rule`; a default is a value the param would accept, already
resolved (a number's default is a float).

`echo` (input `table`, no params) returns its input; it exists only to exercise the machinery.

## Versioning
Why: DESIGN §4.

`schema_version` covers a capability's interface: its input tables and their contracts; its
params' names, types, defaults, and accepted values; its output schema. Every change to it
bumps the version, additions included, so a caller can require what it uses.

[`tests/data/interfaces.json`](../tests/data/interfaces.json) records each interface at each
version, keyed `name/version`: the inputs; the params in declaration order, each with its
type, default, and rule; the output schema of the capability's example call (`EXAMPLES` in
`tests/test_capabilities.py`). A test fails when a capability differs from the record at its
version, has none, or has a record above its version. Records are history: a bump adds one.
- Any difference trips it, a reworded rule included.
- It can't see input-table contracts or checks inside `run`; a `rule` stands in for its
  `valid`. An output type that follows an input's (`echo`'s columns, `feature_id`) is recorded
  for the example's input only. These stay review's.

`host_version` covers the implementation. Every runtime dependency is pinned exactly
([`pyproject.toml`](../pyproject.toml)), so an answer is reproducible per `host_version`: bit
for bit on the same machine and BLAS thread count, to floating-point tolerance across them.
- The compute pins are the libraries that compute answers: scikit-bio, numpy, scipy, pandas,
  patsy, pyarrow, numba, and llvmlite, which compiles numba's kernels. The rest are what they
  need, at the versions `uv.lock` resolves. A test walks the lock from the compute pins, fails
  on a dependency left unpinned or a pin nothing needs, and prints the `dependencies` to paste.
- To move one pin: edit it, `uv lock`, and paste what the test prints if it fails. To
  upgrade: keep only the compute pins, edited; `uv lock --upgrade`; paste what the test
  prints; `uv lock`.
- A security fix in requests, urllib3, or certifi needs a release. They come with scikit-bio,
  for reading URLs, which no capability does.
- Exact versions are not exact files: an index or uv configuration on the installing machine
  can still change what installs.
