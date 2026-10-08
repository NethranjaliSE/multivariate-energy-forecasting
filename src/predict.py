"""Use the saved final model to predict appliance energy use from raw data.

The saved files in ``models/`` are:

    final_ensemble.keras   the three CNN-LSTMs combined; outputs Wh directly
    scaler.joblib          the StandardScaler fitted on the training data
    config.json            the selected feature names and the window length

Usage
-----
    from data_preprocessing import load_raw
    from predict import EnergyPredictor

    df = load_raw("../data/raw/energy_data_set.csv")
    predictor = EnergyPredictor("../models")

    history = predictor.predict(df)     # one prediction per timestamp (backtest)
    next_wh = predictor.predict_next(df)  # forecast for the 10 minutes after the last row

Input requirements
------------------
- The same columns as the original dataset (``rv1`` and ``rv2`` are optional).
- A timestamp index at 10-minute steps, sorted by time, with no gaps.
- At least about 26 hours of history (144 steps for the 24-hour lag plus
  the model's 12-step window) before the first prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from feature_engineering import build_features


class EnergyPredictor:
    """Load the saved ensemble, scaler and settings, and make predictions."""

    def __init__(self, models_dir: str = "../models"):
        models_dir = Path(models_dir)
        self.model = keras.models.load_model(models_dir / "final_ensemble.keras")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        config = json.loads((models_dir / "config.json").read_text())
        self.selected_features = config["selected_features"]
        self.window = config["window"]

    def _windows(self, raw: pd.DataFrame):
        """Build features, scale them and cut them into model input windows."""
        features = build_features(raw)
        columns = list(self.scaler.feature_names_in_)
        scaled = pd.DataFrame(self.scaler.transform(features[columns]),
                              index=features.index, columns=columns)
        X = scaled[self.selected_features].to_numpy(dtype="float32")
        windows = np.lib.stride_tricks.sliding_window_view(X, self.window, axis=0)
        return windows.transpose(0, 2, 1), features.index[self.window - 1:]

    def predict(self, raw: pd.DataFrame) -> pd.Series:
        """Predict Appliances (Wh) for every timestamp with enough history.

        Each prediction uses only information from before its own timestamp,
        so this is a fair backtest over historical data.
        """
        windows, index = self._windows(raw)
        predictions = self.model.predict(windows, verbose=0).ravel()
        return pd.Series(predictions, index=index, name="predicted_Wh")

    def predict_next(self, raw: pd.DataFrame) -> float:
        """Forecast Appliances (Wh) for the 10 minutes after the last row.

        A placeholder row is added for the next timestamp. Its values are
        never used: every feature for time t is built from rows before t,
        so only the real history determines the forecast.
        """
        next_time = raw.index[-1] + pd.Timedelta(minutes=10)
        placeholder = raw.iloc[[-1]].copy()
        placeholder.index = pd.DatetimeIndex([next_time], name=raw.index.name)
        extended = pd.concat([raw, placeholder])
        return float(self.predict(extended).iloc[-1])


def save_predictor(ensemble: keras.Model, scaler, selected_features: list, window: int,
                   models_dir: str = "../models") -> None:
    """Save everything ``EnergyPredictor`` needs into ``models_dir``."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(models_dir / "final_ensemble.keras")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    config = {"selected_features": list(selected_features), "window": int(window)}
    (models_dir / "config.json").write_text(json.dumps(config, indent=2))