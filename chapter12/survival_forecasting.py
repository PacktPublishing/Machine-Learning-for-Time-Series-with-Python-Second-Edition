"""Implement Random Survival Forests for multi-horizon event forecasting."""

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


def simulate_marketing_data(n_samples: int = 3000) -> pd.DataFrame:
    """Generate right-censored marketing telemetry with non-linear risks."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "ad_spend": np.random.gamma(2, 40, n_samples),
            "page_views": np.random.poisson(4, n_samples),
            "channel": np.random.choice([0, 1], n_samples),
        }
    )

    risk = (
        0.015 * df["ad_spend"]
        + 0.3 * df["channel"]
        + 0.05 * df["page_views"]
        + 0.0005 * df["ad_spend"] * df["page_views"]
    )

    time_to_convert = np.random.exponential(1 / np.exp(risk))
    observation_window = 30

    event_observed = time_to_convert <= observation_window
    duration = np.minimum(time_to_convert, observation_window)

    df["duration"] = duration
    df["event"] = event_observed

    return df


def train_survival_forest(
    df: pd.DataFrame, features: list[str]
) -> tuple[RandomSurvivalForest, np.ndarray, pd.DataFrame]:
    """Train a random survival forest model on censored arrays."""
    y = Surv.from_arrays(event=df["event"], time=df["duration"])
    x = df[features]

    model = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(x, y)
    return model, y, x


def forecast_conversion_probabilities(
    model: RandomSurvivalForest, x_inference: pd.DataFrame, horizons: list[int]
) -> pd.DataFrame:
    """Extract multi-horizon conversion probabilities from survival curves."""
    surv_funcs = model.predict_survival_function(x_inference)
    forecasts = []

    for fn in surv_funcs:
        prob_not_converted = fn(horizons)
        prob_converted = 1 - prob_not_converted
        forecasts.append(prob_converted)

    columns = [f"prob_convert_{h}d" for h in horizons]
    return pd.DataFrame(forecasts, columns=columns, index=x_inference.index)


def evaluate_discrimination(
    model: RandomSurvivalForest, x_test: pd.DataFrame, y_test: np.ndarray
) -> float:
    """Calculate the concordance index for rank discrimination."""
    risk_scores = model.predict(x_test)
    c_index, _, _, _, _ = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )
    return c_index
