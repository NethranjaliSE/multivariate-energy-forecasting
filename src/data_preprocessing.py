"""Loading, splitting and scaling for the Appliances energy prediction task.

All splits are chronological and the scaler is fitted on the training
portion only, so no information from the future reaches the model.

Usage
-----
    from data_preprocessing import load_raw, chronological_split, scale_features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "Appliances"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV and return it indexed by timestamp, sorted by time."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def chronological_split(features: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1):
    """Split a feature table by time into train, validation and test parts.

    With the defaults, the first 70% of rows are for training, the next 10%
    for validation (early stopping and tuning) and the last 20% for testing.
    Train plus validation together are the "first 80%" the brief asks for.

    Returns
    -------
    tuple of pd.DataFrame
        (train, validation, test), in time order with no overlap.
    """
    n_rows = len(features)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return features.iloc[:train_end], features.iloc[train_end:val_end], features.iloc[val_end:]


def split_xy(part: pd.DataFrame):
    """Separate a table into features X and the log-transformed target y.

    The target is trained as log(1 + Appliances) because the raw values are
    heavily right-skewed. Use ``to_wh`` to convert predictions back.
    """
    X = part.drop(columns=[TARGET])
    y = np.log1p(part[TARGET])
    return X, y


def to_wh(log_values) -> np.ndarray:
    """Convert log(1 + Wh) values back to Wh."""
    return np.expm1(np.asarray(log_values))


def make_sequences(X_parts, y_parts, window: int):
    """Turn feature tables into overlapping windows for a recurrent model.

    Each sample is the ``window`` most recent feature rows up to and
    including row t, and its label is the target at row t. Every feature
    row is already built only from information available before its own
    timestamp, so a window never contains the value being predicted.

    The parts are joined in time order before the windows are cut, so the
    first validation and test samples can use the rows just before them as
    history. Only past inputs are shared; no target crosses a split.

    Parameters
    ----------
    X_parts : list of pd.DataFrame
        Scaled feature tables in time order, e.g. [X_train, X_val, X_test].
    y_parts : list of pd.Series
        Matching targets, in the same order.
    window : int
        Number of 10-minute steps in each input sequence.

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        One (X, y) pair per part. X has shape (samples, window, features).
        The first ``window - 1`` rows of the first part are dropped because
        they lack a full history.
    """
    X_all = np.concatenate([part.to_numpy(dtype="float32") for part in X_parts])
    y_all = np.concatenate([part.to_numpy(dtype="float32") for part in y_parts])

    # Windows ending at every row from (window - 1) onwards.
    windows = np.lib.stride_tricks.sliding_window_view(X_all, window, axis=0)
    windows = windows.transpose(0, 2, 1)  # (samples, window, features)
    end_rows = np.arange(window - 1, len(X_all))

    results = []
    start = 0
    for part in X_parts:
        stop = start + len(part)
        mask = (end_rows >= start) & (end_rows < stop)
        results.append((np.ascontiguousarray(windows[mask]), y_all[end_rows[mask]]))
        start = stop
    return results


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Standardise features to zero mean and unit variance.

    The scaler is fitted on the training data only and then applied to the
    validation and test data, which prevents leakage.

    Returns
    -------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    scaled = [
        pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns),
        pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns),
        pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns),
    ]
    return scaled[0], scaled[1], scaled[2], scaler