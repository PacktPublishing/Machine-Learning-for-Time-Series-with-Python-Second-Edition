"""Marketing conversion forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def simulate_campaign_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic right-censored marketing conversion data."""
    np.random.seed(42)
    channel = np.random.choice(["Search", "Social"], size=n_samples)
    hazard = np.where(channel == "Search", 0.08, 0.04)
    time_to_event = np.random.exponential(1 / hazard)
    observation_window = 30
    observed = time_to_event <= observation_window
    duration = np.minimum(time_to_event, observation_window)
    return pd.DataFrame(
        {"duration": duration, "event": observed.astype(int), "channel": channel}
    )


def plot_kaplan_meier_baseline(df: pd.DataFrame) -> KaplanMeierFitter:
    """Fit and visualize Kaplan-Meier survival curves by channel."""
    kmf = KaplanMeierFitter()
    plt.figure()
    for ch in df["channel"].unique():
        mask = df["channel"] == ch
        kmf.fit(df[mask]["duration"], event_observed=df[mask]["event"], label=ch)
        kmf.plot_survival_function()
    plt.title("Time to Conversion by Channel")
    plt.xlabel("Days Since Acquisition")
    plt.ylabel("Probability Not Yet Converted")
    plt.show()
    return kmf


def generate_probabilistic_forecast(
    df: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Train Random Survival Forest and extract discrete forecasts."""
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["duration"])
    x_encoded = pd.get_dummies(df[["channel"]], drop_first=True)
    rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
    rsf.fit(x_encoded, y)
    surv_funcs = rsf.predict_survival_function(x_encoded)

    forecasts = []
    for fn in surv_funcs:
        surv_probs = [fn(t) for t in horizons]
        conv_probs = [1 - p for p in surv_probs]
        forecasts.append(conv_probs)

    return pd.DataFrame(forecasts, columns=[f"day_{h}" for h in horizons])
