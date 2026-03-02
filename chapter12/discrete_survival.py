"""Implement discrete hazard forecasting using panel expansion."""

import pandas as pd
from lightgbm import LGBMClassifier
from mlforecast import MLForecast


def expand_survival_to_panel(df: pd.DataFrame, max_horizon: int = 30) -> pd.DataFrame:
    """Transform right-censored records into discrete-time panels."""
    rows = []
    for idx, row in df.iterrows():
        duration = int(row["duration"])
        event = row["event"]
        limit = min(duration, max_horizon)

        for t in range(1, limit + 1):
            is_event = 1 if (event == 1 and t == duration) else 0
            rows.append(
                {
                    "unique_id": idx,
                    "ds": t,
                    "event": is_event,
                    "ad_spend": row.get("ad_spend", 0),
                    "page_views": row.get("page_views", 0),
                    "channel": row.get("channel", 0),
                }
            )

    return pd.DataFrame(rows)


def train_discrete_hazard_model(panel_df: pd.DataFrame) -> MLForecast:
    """Train a gradient boosted classifier on the hazard panel."""
    model = MLForecast(
        models=[LGBMClassifier(random_state=42)],
        freq=1,
        lags=[1, 2, 3],
    )

    model.fit(
        df=panel_df,
        id_col="unique_id",
        time_col="ds",
        target_col="event",
        static_features=["ad_spend", "page_views", "channel"],
    )

    return model


def forecast_discrete_survival(model: MLForecast, horizon: int = 30) -> pd.DataFrame:
    """Calculate cumulative survival from discrete hazard predictions."""
    hazard_forecast = model.predict(horizon)

    survival_probability = 1 - hazard_forecast["LGBMClassifier"]
    hazard_forecast["survival"] = survival_probability.groupby(
        hazard_forecast["unique_id"]
    ).cumprod()

    return hazard_forecast
