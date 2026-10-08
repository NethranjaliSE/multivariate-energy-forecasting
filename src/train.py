"""Training, evaluation and hyperparameter search for the sequence models.

Usage
-----
    from train import SEARCH_SPACE, sample_configs, run_trial
"""

import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing import make_sequences, to_wh
from model import build_cnn_lstm, default_callbacks

# Candidate values for the CNN-LSTM random search.
SEARCH_SPACE = {
    "window": [6, 12, 24],           # history length in 10-minute steps (1, 2 or 4 hours)
    "filters": [16, 32, 64],         # convolution filters
    "kernel_size": [2, 3, 5],        # convolution width in steps
    "units": [32, 64, 128],          # LSTM units
    "dense_units": [16, 32, 64],     # neurons in the dense layer
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128],
}

# The configuration used before tuning, for the before/after comparison.
DEFAULT_CONFIG = {
    "window": 12, "filters": 32, "kernel_size": 3, "units": 64,
    "dense_units": 32, "dropout": 0.2, "learning_rate": 1e-3, "batch_size": 64,
}


def sample_configs(space: dict, n_trials: int, seed: int = 0) -> list:
    """Draw ``n_trials`` distinct random configurations from ``space``.

    Random search is used instead of a full grid: this space has 8,748
    combinations, far too many to train, and random search tends to find
    good settings with few trials because only some hyperparameters matter.
    """
    rng = random.Random(seed)
    configs, seen = [], set()
    while len(configs) < n_trials:
        config = {name: rng.choice(values) for name, values in space.items()}
        key = tuple(config.values())
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def run_trial(config: dict, X_parts: list, y_parts: list, y_mean: float, y_std: float,
              seed: int = 42, max_epochs: int = 150, patience: int = 10):
    """Train one CNN-LSTM with ``config`` and score it on the validation set.

    Parameters
    ----------
    config : dict
        Hyperparameters, with the same keys as ``SEARCH_SPACE``.
    X_parts : list of pd.DataFrame
        [X_train, X_val, X_test], scaled and limited to the selected features.
    y_parts : list of pd.Series
        [y_train, y_val, y_test] as log(1 + Wh), not yet standardised.
    y_mean, y_std : float
        Training-set statistics used to standardise the log target.
    seed : int
        Random seed, so results can be repeated.

    Returns
    -------
    tuple
        (metrics dict, trained model, sequences) where sequences is the
        output of ``make_sequences`` for this window, ready for testing.
    """
    standardised = [(y - y_mean) / y_std for y in y_parts]
    sequences = make_sequences(X_parts, standardised, config["window"])
    (X_train, y_train), (X_val, y_val), _ = sequences

    tf.keras.utils.set_random_seed(seed)
    model = build_cnn_lstm(
        config["window"], X_train.shape[2],
        filters=config["filters"], kernel_size=config["kernel_size"],
        units=config["units"], dense_units=config["dense_units"],
        dropout=config["dropout"], learning_rate=config["learning_rate"],
    )
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs, batch_size=config["batch_size"],
        callbacks=default_callbacks(patience), verbose=0,
    )

    val_actual = to_wh(y_val * y_std + y_mean)
    val_pred = to_wh(model.predict(X_val, verbose=0).ravel() * y_std + y_mean)
    metrics = {
        **config,
        "seed": seed,
        "epochs": len(history.history["loss"]),
        "seconds": round(time.time() - start),
        "val MAE": mean_absolute_error(val_actual, val_pred),
        "val RMSE": float(np.sqrt(mean_squared_error(val_actual, val_pred))),
    }
    return metrics, model, sequences