"""Chapter 9 benchmark: Seasonal Naive vs. LightGBM vs. zero-shot Chronos-Tiny.

Holds out the last 24 months of AirPassengers and scores all three with MAPE,
sMAPE, and MASE. Demonstrates the "baseline first" branch of the chapter's
decision flowchart.

Run from the repo root:
    python chapter9/baseline_benchmark.py
"""
# lightgbm before torch on macOS to avoid a libomp double-load segfault.
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

np.random.seed(0)
torch.manual_seed(0)

URL = (
    "https://raw.githubusercontent.com/AileenNielsen/"
    "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)
df = pd.read_csv(URL)
y = df["#Passengers"].values.astype(float)

H = 24       # forecast horizon (last 24 months held out)
M = 12       # seasonal period
y_train, y_test = y[:-H], y[-H:]


def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y)) * 100


def smape(y, yhat):
    return np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100


def mase(y, yhat, y_train, m=12):
    naive_err = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y - yhat)) / naive_err


seasonal_naive = np.tile(y_train[-M:], H // M + 1)[:H]


def make_features(values, lags=(1, 2, 3, 12, 24), month_period=12):
    X, y_out, months = [], [], []
    for i in range(max(lags), len(values)):
        X.append([values[i - L] for L in lags])
        y_out.append(values[i])
        months.append(i % month_period)
    X = np.array(X)
    months_oh = np.eye(month_period)[months]
    return np.hstack([X, months_oh]), np.array(y_out)


X_train, yt = make_features(y_train)
booster = lgb.train(
    {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 4,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": 0,
    },
    lgb.Dataset(X_train, label=yt),
    num_boost_round=400,
)

lags = (1, 2, 3, 12, 24)
history = list(y_train)
gbm_forecast = []
for _ in range(H):
    idx = len(history)
    row_lags = [history[idx - L] for L in lags]
    months_oh = np.eye(12)[idx % 12]
    x_row = np.concatenate([row_lags, months_oh]).reshape(1, -1)
    pred = float(booster.predict(x_row)[0])
    gbm_forecast.append(pred)
    history.append(pred)
gbm_forecast = np.array(gbm_forecast)


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)
samples = pipeline.predict(
    inputs=torch.tensor(y_train, dtype=torch.float32),
    prediction_length=H,
    num_samples=100,
)
chronos_forecast = samples.numpy().squeeze(0).mean(axis=0)


rows = [
    {
        "Model": name,
        "MAPE": round(mape(y_test, yhat), 2),
        "sMAPE": round(smape(y_test, yhat), 2),
        "MASE": round(mase(y_test, yhat, y_train, m=M), 3),
    }
    for name, yhat in [
        ("Seasonal Naive", seasonal_naive),
        ("LightGBM (lag + month)", gbm_forecast),
        ("Chronos-Tiny zero-shot", chronos_forecast),
    ]
]

print(pd.DataFrame(rows).to_string(index=False))
