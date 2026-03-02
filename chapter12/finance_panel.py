"""Financial risk discrete panels."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def simulate_loan_portfolio(n_samples: int = 1500) -> pd.DataFrame:
    """Generate right-censored loan duration data with credit scores."""
    np.random.seed(0)
    credit_score = np.random.normal(650, 50, n_samples)
    baseline_hazard = 0.02
    risk_multiplier = np.exp(-(credit_score - 650) / 100)
    hazard = baseline_hazard * risk_multiplier
    time_to_default = np.random.exponential(1 / hazard)
    observation_window = 60
    observed = time_to_default <= observation_window
    duration = np.minimum(time_to_default, observation_window)
    return pd.DataFrame(
        {
            "id": range(n_samples),
            "duration": duration,
            "event": observed.astype(int),
            "credit_score": credit_score,
        }
    )


def fit_cox_hazard_model(df: pd.DataFrame) -> CoxPHFitter:
    """Fit a Cox model to determine hazard ratios for credit scores."""
    cph = CoxPHFitter()
    df_fit = df.drop(columns=["id"])
    cph.fit(df_fit, duration_col="duration", event_col="event")
    cph.print_summary()
    return cph


def expand_to_longitudinal_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Convert right-censored records into person-period panels."""
    panel_rows = []
    for _, row in df.iterrows():
        t_max = int(np.ceil(row["duration"]))
        for t in range(1, t_max + 1):
            is_event = 1 if (t == t_max and row["event"] == 1) else 0
            panel_rows.append(
                {
                    "id": row["id"],
                    "time_period": t,
                    "credit_score": row["credit_score"],
                    "event": is_event,
                }
            )
    return pd.DataFrame(panel_rows)
