"""Sanity-test the two runnable chapter-9 examples as written.

- TimesFM synthetic-data demo (Google TimesFM section)
- Chronos AirPassengers demo (Amazon Chronos section)

Mirrors the chapter's code as faithfully as possible, including the recent fixes
(point_forecast[0]/[1] indexing, dtype=torch.float32, inputs= kwarg).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import numpy as np
import pandas as pd
import torch

np.random.seed(0)
torch.manual_seed(0)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


# ---------- TimesFM synthetic demo ----------
section("TimesFM: synthetic seasonal + trend, zero-shot")

import timesfm

# Chapter line 240ish: create + compile the 200M model.
# Note: torch_compile=True from the chapter is known to be slow on CPU on first call,
# so we leave it off here to keep the smoke test fast. The chapter shows it as an
# optional optimization flag.
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
tfm.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=96,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
))
print("TimesFM compiled.")

# Chapter synthetic data (lines ~286)
t = np.linspace(0, 50, 500)
series_seasonal = 10 * np.sin(t) + np.random.normal(0, 1, 500) + 20
series_trend = np.linspace(10, 100, 500) + np.random.normal(0, 2, 500)

# IMPORTANT: TimesFM-2.5 silently returns NaN on float64; cast to float32.
input_data = [series_seasonal.astype(np.float32), series_trend.astype(np.float32)]

point_forecast, quantile_forecast = tfm.forecast(horizon=96, inputs=input_data)
print(
    f"point_forecast shape: {point_forecast.shape}, "
    f"NaN? {np.isnan(point_forecast).any()}"
)
print(
    f"quantile_forecast shape: {quantile_forecast.shape}, "
    f"NaN? {np.isnan(quantile_forecast).any()}"
)
print(f"Seasonal forecast first 5: {point_forecast[0][:5]}")
print(f"Trend forecast first 5: {point_forecast[1][:5]}")


# ---------- Chronos AirPassengers demo ----------
section("Chronos: AirPassengers, zero-shot")

from chronos import ChronosPipeline

# Chapter currently uses the Large model on CUDA. For CPU testing we drop to Tiny.
# Important: dtype=torch.float32 is the fix we just landed; older code used
# the NumPy float64 default which Chronos can silently mishandle.
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)
print("Chronos pipeline loaded.")

url = (
    "https://raw.githubusercontent.com/AileenNielsen/"
    "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)
df = pd.read_csv(url)
context_tensor = torch.tensor(df["#Passengers"].values, dtype=torch.float32)

# Fix verified: inputs=... (older releases used context=...)
forecast = pipeline.predict(
    inputs=context_tensor,
    prediction_length=24,
    num_samples=100,
    limit_prediction_length=False,
)
forecast_np = forecast.numpy()
print(f"forecast shape: {forecast_np.shape}, NaN? {np.isnan(forecast_np).any()}")

p10 = np.quantile(forecast_np, 0.1, axis=1)
p50 = np.quantile(forecast_np, 0.5, axis=1)
p90 = np.quantile(forecast_np, 0.9, axis=1)
print(f"p10 first 3: {p10.squeeze()[:3]}")
print(f"p50 first 3: {p50.squeeze()[:3]}")
print(f"p90 first 3: {p90.squeeze()[:3]}")

section("All chapter-9 runnable examples passed.")
