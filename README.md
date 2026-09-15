# Machine-Learning-for-Time-Series-with-Python-Second-Edition
Machine Learning for Time-Series with Python, Second Edition - Published by Packt

## Environments

Most chapters run in the main environment defined by `pyproject.toml`. A few
notebooks need extra dependencies kept out of that file:

- **Chapter 9** (`chapter9/README.md`): `TimesFM` needs Go and Rust toolchains
  to build `wandb` from source.
- **Chapter 11** (`chapter11/README.md`): `Graph WaveNet.ipynb` and
  `Neural ODEs.ipynb` need `torch` and `torchdiffeq`. Run them in a dedicated
  `ts_advanced` conda environment; see that chapter's README for the recipe.

## TODOs

### Migrate to numpy 2.x

`pyproject.toml` currently pins `numpy>=1.26.0,<2.0.0`. The cap is conservative,
not forced by any downstream library. Migration should be straightforward:

- **Why we are still on numpy 1.x:** torch 2.2.2 (the version installed via the
  current pin chain) was built against numpy 1.x. numpy 2 ABI support landed in
  torch 2.3.0.
- **What blocks the migration:** only torch itself. Audited 2026-05-11; no other
  forecasting library in `pyproject.toml` caps numpy below 2.

Compatibility audit (numpy 2 caps across the env):

| Package              | numpy constraint     | numpy 2 OK? |
|----------------------|----------------------|-------------|
| chronos-forecasting  | `<3,>=1.21`          | yes         |
| sktime               | `<2.4,>=1.21`        | yes         |
| arch                 | `<3,>=1.22.3`        | yes         |
| statsmodels          | `<3,>=1.22.3`        | yes         |
| mlflow               | `<3`                 | yes         |
| All other forecast libs (statsforecast, skforecast, hierarchicalforecast, neuralforecast, darts, tsfresh, pmdarima, river, nannyml, shap, xgboost, lightgbm) | unbounded or `>=` only | yes |

Compatibility audit (torch caps that constrain the upgrade):

| Package           | torch constraint   |
|-------------------|--------------------|
| chronos-forecasting | `>=2.2,<3`       |
| darts             | `>=2.0.0`          |
| neuralforecast    | `>=2.0.0,<=2.6.0`  |
| pytorch-lightning | `>=2.1.0`          |
| lightning         | `>=2.1.0,<4.0`     |

**Safe torch window for the migration: 2.3.0 - 2.6.0** (lower = first
numpy-2-compatible torch, upper = neuralforecast cap).

**Migration recipe:**

1. Bump `pyproject.toml`: change `numpy>=1.26.0,<2.0.0` to `numpy>=2.0.0,<3.0.0`,
   and pin torch in the `2.3.0 - 2.6.0` range.
2. Extend the GitHub Actions matrix beyond `test-chapter6.yml` to add at least
   chapters 7 and 9 (deep-learning heavy chapters most likely to surface
   torch/numpy ABI issues) before flipping the cap.
3. Run all chapter notebooks locally in a fresh env, fix any breakages, then
   merge.
