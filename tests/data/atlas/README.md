# HITChip Atlas fixtures

Verbatim from scikit-bio 0.7.4, `skbio/stats/composition/tests/data/`
(<https://github.com/scikit-bio/scikit-bio/tree/0.7.4/skbio/stats/composition/tests/data>),
BSD-3-Clause, copyright the scikit-bio development team.

| File | Holds |
|---|---|
| `atlas_feature_table.csv` | 300 samples × 21 families, counts |
| `atlas_meta_data.csv` | `age`, `region` (4 levels), `bmi` (3 levels) |
| `atlas_ancombc_main.tsv` | R ANCOMBC 2.13.2, primary results |
| `atlas_ancombc_global.tsv` | R ANCOMBC 2.13.2, global test |

sha256:
```
6a1a899472f449460f9cbb914d7de729409f82f8b745cdcbb4537956d40f2745  atlas_feature_table.csv
69a945e91d2e578de942e710b8e8550895b745f2c57ac4cadc07378b4e07f6ec  atlas_meta_data.csv
fbd49be6e922ec7708fba58b3d94e4165f73a83290bb36620725cbd72994e6c1  atlas_ancombc_main.tsv
b653ec9525269461df88da81e432282ec4555e3640668d65a00febc69480ba14  atlas_ancombc_global.tsv
```

The data: Lahti et al., "Tipping elements in the human intestinal ecosystem", Nature
Communications 5, 4344 (2014), deposited at Dryad under CC0 1.0
(<https://doi.org/10.5061/dryad.pk75d>). scikit-bio subset it following the ANCOM-BC tutorial:
taxa aggregated to family, 300 samples drawn at random.

scikit-bio made the reference outputs with R ANCOMBC 2.13.2, reformatted to its Python
output's layout, by this script (from its `test_ancombc.py`):

```R
library(ANCOMBC)

set.seed(42)

table <- read.csv("atlas_feature_table.csv", row.names = 1)
meta <- read.csv("atlas_meta_data.csv", row.names = 1)
meta$bmi <- factor(meta$bmi, levels = c("lean", "overweight", "obese"))

res_bc <- ancombc(
    data = table,
    taxa_are_rows = FALSE,
    meta_data = meta,
    formula = "age + region + bmi",
    group = "bmi",
    p_adj_method = "holm",
    prv_cut = 0,
    lib_cut = 0,
    pseudo = 1,
    tol = 1e-5,
    max_iter = 100,
    conserve = FALSE,
    alpha = 0.05,
    global = TRUE,
    struc_zero = FALSE,
    neg_lb = FALSE,
    n_cl = 1,
    verbose = FALSE
)

write.csv(res_bc$res, "atlas_ancombc_main.csv", row.names = FALSE)
write.csv(res_bc$res_global, "atlas_ancombc_global.csv", row.names = FALSE)
```
