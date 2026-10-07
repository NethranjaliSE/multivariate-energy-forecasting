"""Feature engineering for the Appliances energy prediction task.

Forecast setup
--------------
The target is ``Appliances`` at time t. Every feature must be known
10 minutes earlier (at time t-1) or be a calendar fact about time t.
To guarantee this, all sensor readings and all consumption-based
features are shifted by at least one step. This prevents data leakage.

Usage
-----
    from feature_engineering import build_features
    features = build_features(df)   # df has a DatetimeIndex at 10-minute steps
"""

import numpy as np
import pandas as pd

TARGET = "Appliances"

# Random noise columns included in the dataset on purpose; they carry no signal.
NOISE_COLUMNS = ["rv1", "rv2"]

# Lags in 10-minute steps, chosen from the autocorrelation analysis:
# 10, 20, 30, 60, 120 minutes (strong short-term memory) and 24 hours (daily cycle).
LAG_STEPS = [1, 2, 3, 6, 12, 144]

# Rolling windows in 10-minute steps.
ROLLING_WINDOWS = {"1h": 6, "3h": 18}

# Indoor sensors. T6 / RH_6 are excluded because that sensor is outside the building.
INDOOR_TEMP_COLUMNS = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_HUMIDITY_COLUMNS = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]

# Belgian public holidays inside the data range (the house is in Belgium).
BELGIAN_HOLIDAYS_2016 = pd.to_datetime(
    [
        "2016-03-28",  # Easter Monday
        "2016-05-01",  # Labour Day
        "2016-05-05",  # Ascension Day
        "2016-05-16",  # Whit Monday
    ]
)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the timestamp index.

    Hour of day is encoded with sine and cosine so that 23:50 and 00:00
    are close together, which a plain 0-23 number would not capture.
    Month is deliberately left out: the data covers only 4.5 months, so a
    chronological test set would contain months barely seen in training.
    """
    out = df.copy()
    idx = out.index

    out["NSM"] = idx.hour * 3600 + idx.minute * 60 + idx.second  # seconds since midnight
    hour_fraction = out["NSM"] / 86400.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    out["hour_cos"] = np.cos(2 * np.pi * hour_fraction)

    out["day_of_week"] = idx.dayofweek  # 0 = Monday ... 6 = Sunday
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = idx.normalize().isin(BELGIAN_HOLIDAYS_2016).astype(int)

    # Interaction: lets a model learn a different daily shape at weekends.
    out["weekend_hour_sin"] = out["is_weekend"] * out["hour_sin"]
    out["weekend_hour_cos"] = out["is_weekend"] * out["hour_cos"]
    return out


def shift_sensor_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Shift every sensor column one step back so it is known before time t.

    Without this, the model would use the temperature and humidity measured
    during the same 10 minutes it is trying to predict.
    """
    out = df.copy()
    sensor_columns = [c for c in out.columns if c != TARGET]
    out[sensor_columns] = out[sensor_columns].shift(1)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add past consumption values at the lags in ``LAG_STEPS``."""
    out = df.copy()
    for steps in LAG_STEPS:
        out[f"app_lag_{steps}"] = out[TARGET].shift(steps)
    # Short-term direction of change (is usage rising or falling?).
    out["app_diff_1"] = out[TARGET].shift(1) - out[TARGET].shift(2)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics of past consumption.

    The series is shifted by one step first, so a window ending at t-1
    never includes the value being predicted.
    """
    out = df.copy()
    past = out[TARGET].shift(1)
    for label, window in ROLLING_WINDOWS.items():
        out[f"app_roll_mean_{label}"] = past.rolling(window).mean()
        out[f"app_roll_std_{label}"] = past.rolling(window).std()
        out[f"app_roll_max_{label}"] = past.rolling(window).max()
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined temperature and humidity features.

    Must be called after ``shift_sensor_readings`` so the inputs are already
    lagged by one step.
    """
    out = df.copy()
    out["indoor_temp_mean"] = out[INDOOR_TEMP_COLUMNS].mean(axis=1)
    out["indoor_rh_mean"] = out[INDOOR_HUMIDITY_COLUMNS].mean(axis=1)
    # How much warmer the house is than outside (a proxy for heating demand).
    out["temp_gap_in_out"] = out["indoor_temp_mean"] - out["T_out"]
    # Temperature x humidity interaction.
    out["temp_x_rh"] = out["indoor_temp_mean"] * out["indoor_rh_mean"]
    # Kitchen humidity relative to the rest of the house (cooking raises it).
    out["kitchen_rh_excess"] = out["RH_1"] - out["indoor_rh_mean"]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and return a model-ready table.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data indexed by timestamp at 10-minute intervals, sorted by time.

    Returns
    -------
    pd.DataFrame
        The target column plus all features. Rows at the start that lack
        enough history for the longest lag are dropped.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("The index must be sorted by time before building features.")

    out = df.drop(columns=[c for c in NOISE_COLUMNS if c in df.columns])
    out = shift_sensor_readings(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_interaction_features(out)
    out = add_time_features(out)
    return out.dropna()