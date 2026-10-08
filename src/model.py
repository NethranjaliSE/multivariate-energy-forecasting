"""Deep learning models for the Appliances energy prediction task.

Every model takes input of shape (window, n_features) and predicts one
value: the standardised log(1 + Appliances) for the next 10-minute interval.

All models share the same output head and training setup, so differences
in results come from the sequence layers alone:

    sequence layers  ->  Dropout  ->  Dense(dense_units, ReLU)  ->  Dense(1, linear)

Shared choices
--------------
- Recurrent layers keep their default tanh activation, which bounds the
  internal state and is what TensorFlow's fast implementations expect.
- ReLU in the dense layer adds non-linearity without saturating.
- The output layer is linear because this is a regression problem.
- Dropout reduces overfitting.
- Adam adapts the learning rate for each weight and works well with
  little tuning; mean squared error is the standard regression loss.

Usage
-----
    from model import build_lstm, build_gru, build_cnn_lstm, build_stacked_lstm
    model = build_lstm(window=12, n_features=10)
"""

from tensorflow import keras
from tensorflow.keras import layers


def _finish(name: str, inputs, sequence_output, dense_units: int, dropout: float,
            learning_rate: float) -> keras.Model:
    """Add the shared output head to a sequence block and compile the model."""
    x = layers.Dropout(dropout)(sequence_output)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
    return model


def build_lstm(window: int, n_features: int, units: int = 64, dense_units: int = 32,
               dropout: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    """Single LSTM layer: the main model.

    An LSTM has input, forget and output gates that decide what to keep in
    its memory, which lets it learn which past steps matter.
    """
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(units)(inputs)
    return _finish("lstm", inputs, x, dense_units, dropout, learning_rate)


def build_gru(window: int, n_features: int, units: int = 64, dense_units: int = 32,
              dropout: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    """Single GRU layer.

    A GRU merges the LSTM's gates into two (update and reset), so it has
    about 25% fewer parameters and often trains faster for similar accuracy.
    """
    inputs = keras.Input(shape=(window, n_features))
    x = layers.GRU(units)(inputs)
    return _finish("gru", inputs, x, dense_units, dropout, learning_rate)


def build_cnn_lstm(window: int, n_features: int, filters: int = 32, kernel_size: int = 3,
                   units: int = 64, dense_units: int = 32, dropout: float = 0.2,
                   learning_rate: float = 1e-3) -> keras.Model:
    """1D convolution followed by an LSTM (CNN-LSTM hybrid).

    The convolution scans short stretches of the window (3 steps = 30
    minutes by default) and learns local patterns such as the start of a
    spike. The LSTM then models how those patterns develop over time.
    Causal padding stops each convolution output from looking at later
    steps in the window.
    """
    inputs = keras.Input(shape=(window, n_features))
    x = layers.Conv1D(filters, kernel_size, padding="causal", activation="relu")(inputs)
    x = layers.LSTM(units)(x)
    return _finish("cnn_lstm", inputs, x, dense_units, dropout, learning_rate)


def build_stacked_lstm(window: int, n_features: int, units: int = 64, second_units: int = 32,
                       dense_units: int = 32, dropout: float = 0.2,
                       learning_rate: float = 1e-3) -> keras.Model:
    """Two LSTM layers, the first passing its full sequence to the second.

    Tests whether extra depth helps. Dropout between the layers limits
    overfitting from the added parameters.
    """
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(second_units)(x)
    return _finish("stacked_lstm", inputs, x, dense_units, dropout, learning_rate)


def default_callbacks(patience: int = 10) -> list:
    """Early stopping and learning-rate reduction, both based on validation loss.

    Early stopping halts training when the validation loss stops improving
    and restores the best weights seen. The learning rate is halved when
    progress stalls, which often lets training settle into a better minimum.
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience // 2, min_lr=1e-5),
    ]