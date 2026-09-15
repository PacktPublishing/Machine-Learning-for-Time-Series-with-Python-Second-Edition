# Chapter 5: Feature Engineering and Tree-Based Models

This chapter uses the **M5 Forecasting Accuracy** dataset: 30,490 daily time
series of unit sales for 3,049 Walmart products across 10 stores in three US
states (CA, TX, WI). The data spans about five years and includes calendar
events, SNAP food-benefit days, and weekly sell prices, which makes it a rich
real-world panel for the chapter's feature engineering and gradient-boosting
workflow.

## Getting the data

The M5 dataset is hosted on Zenodo under a CC-BY 4.0 license, so the notebooks
do not redistribute it directly. Run the download script before the notebooks:

```bash
python download_m5.py
```

The script downloads `m5-forecasting-accuracy.zip` (~48 MB) from Zenodo,
verifies its MD5, and extracts the CSV files into the chapter directory:

- `calendar.csv` — date metadata, events, SNAP flags
- `sales_train_validation.csv` — wide-format daily sales (`d_1`…`d_1913`)
- `sales_train_evaluation.csv` — extended training set used for the final
  evaluation phase of the original competition
- `sell_prices.csv` — weekly sell price per (store, item)
- `sample_submission.csv` — competition submission template

Pass `--dest <dir>` to put the files somewhere else.

## Notebooks

1. `01_data_prep_and_features.ipynb` — load, melt, merge, and engineer the
   feature set used by the rest of the chapter.
2. `02_forecast_model_lightgbm.ipynb` — train and tune a LightGBM model with
   Optuna, including the inverse-transform step for euro-scale evaluation.
3. `03_advanced_and_automated.ipynb` — sktime reduction, mlforecast,
   tsfresh, and classical decomposition as comparison patterns.

## Attribution

> Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).
> *M5 Forecasting Accuracy Dataset.* Zenodo.
> [https://doi.org/10.5281/zenodo.12636070](https://doi.org/10.5281/zenodo.12636070)

Licensed under
[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
