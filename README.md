# Appliance Energy Forecasting with Deep Learning

Multivariate time-series forecasting of household appliance energy use. A CNN-LSTM ensemble predicts the energy used by a house's appliances in the **next 10 minutes**, using only information available before that interval.

[![Live demo on Hugging Face](https://img.shields.io/badge/Live%20demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/NethranjaliSE/appliance-energy-forecasting)
[![Model on Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-blue)](https://huggingface.co/NethranjaliSE/appliance-energy-cnn-lstm)

- **Live demo:** [huggingface.co/spaces/NethranjaliSE/appliance-energy-forecasting](https://huggingface.co/spaces/NethranjaliSE/appliance-energy-forecasting) replays the held-out test period and compares each forecast with what was actually measured.
- **Trained model:** [huggingface.co/NethranjaliSE/appliance-energy-cnn-lstm](https://huggingface.co/NethranjaliSE/appliance-energy-cnn-lstm)

## Results

Test set: 30 April to 27 May 2016, the last 20% of the data in time order, never used for training or model selection.

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Naive (repeat the last reading) | 26.05 | 65.19 | 21.86 | 0.45 |
| Linear Regression | 27.26 | 66.49 | 22.12 | 0.43 |
| Random Forest | 26.51 | 58.30 | 23.19 | 0.56 |
| LSTM | 23.41 | 56.78 | 19.38 | 0.58 |
| CNN-LSTM (before optimisation) | 23.70 | 57.55 | 19.72 | 0.57 |
| **CNN-LSTM ensemble (final model)** | **22.80** | **55.71** | **19.01** | **0.60** |

Compared with the naive baseline, the final model reduces MAE by 12.5% and RMSE by 14.5%. It follows the household's daily routine closely; its main limitation is sudden spikes, which it detects about 10 minutes late and underestimates.

## Dataset

[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), UCI Machine Learning Repository (CC BY 4.0).

- 19,735 readings at 10-minute intervals, 11 January to 27 May 2016, from a low-energy house in Belgium
- Target: `Appliances`, energy use in Wh
- Inputs: lighting energy, temperature and humidity in nine areas of the house, and outdoor weather from a nearby airport
- No missing values, no duplicate timestamps and no gaps in the timeline

The file is included at `data/raw/energy_data_set.csv`.

## Project structure

```
├── data/
│   ├── raw/energy_data_set.csv       # original dataset
│   └── processed/features.csv        # engineered feature table
├── notebooks/
│   └── EDA.ipynb                     # full workflow: EDA, features, models, tuning, final model
├── src/
│   ├── data_preprocessing.py         # loading, chronological split, scaling, sequence windows
│   ├── feature_engineering.py        # lag, rolling, time, interaction and holiday features
│   ├── model.py                      # LSTM, GRU, CNN-LSTM, stacked LSTM and the ensemble
│   ├── train.py                      # training, validation scoring and random search
│   └── predict.py                    # load the saved model and make predictions
├── models/
│   ├── final_ensemble.keras          # final model: three CNN-LSTMs averaged, outputs Wh
│   ├── scaler.joblib                 # feature scaler fitted on the training data
│   ├── config.json                   # selected features and window length
│   └── cnn_lstm_seed{42,7,123}.keras # the three ensemble members
├── reports/
│   ├── report.pdf                    # project report
│   ├── random_search_results.csv     # all hyperparameter search trials
│   └── final_results.csv             # final test-set comparison
├── requirements.txt
└── README.md
```

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/NethranjaliSE/multivariate-energy-forecasting.git
cd multivariate-energy-forecasting
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate            # Windows (Command Prompt)
venv\Scripts\Activate.ps1        # Windows (PowerShell)
source venv/bin/activate         # macOS or Linux
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

| Library | Version | Used for |
|---|---|---|
| pandas | 3.0.6 | data handling |
| NumPy | 2.4.6 | numerical work |
| Matplotlib | 3.11.2 | plots |
| Seaborn | 0.13.2 | statistical plots |
| scikit-learn | 1.9.1 | scaling, baselines, metrics |
| SciPy | 1.17.1 | scikit-learn dependency |
| TensorFlow (Keras 3) | 2.21.0 | deep learning models |
| Jupyter, ipykernel | 1.1.1, 7.4.0 | running the notebook |

Training runs on the CPU; no GPU is needed.

## How to run

1. Start Jupyter with `jupyter notebook`, or open the project folder in VS Code and select the `venv` kernel.
2. Open `notebooks/EDA.ipynb`.
3. Run all cells from top to bottom (**Restart**, then **Run All**).

A full run takes about 75 minutes on a laptop CPU. Most of that time is the experiments, which train many models:

| Notebook section | Approximate time |
|---|---|
| Data checks, EDA, features, split, baselines, feature selection | 3 minutes |
| LSTM | 1 minute |
| Architecture comparison (4 models × 3 seeds) | 15 minutes |
| Random search (20 configurations) | 30 minutes |
| Re-check of the top 3 configurations | 8 minutes |
| Dropout experiment and final ensemble | 18 minutes |

The notebook saves the engineered features to `data/processed/`, the search and final results to `reports/`, and the trained models to `models/`.

## Using the trained model

The saved model can make predictions without retraining. `predict.py` repeats all the preparation steps (features, scaling, feature selection, input windows) and returns predictions in Wh.

```python
import sys
sys.path.append("src")

from data_preprocessing import load_raw
from predict import EnergyPredictor

df = load_raw("data/raw/energy_data_set.csv")
predictor = EnergyPredictor("models")

predictor.predict(df)       # a prediction for every 10-minute step with enough history
predictor.predict_next(df)  # the forecast for the 10 minutes after the last reading
```

The input needs the same columns as the original dataset and at least about 26 hours of history: 24 hours for the daily lag feature plus the model's 2-hour window.

Models are saved in the native Keras format (`.keras`), the recommended format in Keras 3, rather than the older `.h5` format.

## Approach

### Task

One-step-ahead forecasting: predict `Appliances` for the next 10-minute interval. Every feature uses only information from before that interval, so the model never sees the value it is predicting.

### Data preprocessing

- **Missing values:** none found; no imputation needed.
- **Outliers:** 10.8% of readings are above the IQR upper bound of 175 Wh. They are real high-use periods, not errors, so they were kept. The target is modelled as `log(1 + Appliances)` to reduce its heavy right skew, then standardised using training-set statistics. All metrics are reported in Wh.
- **Scaling:** standardisation, fitted on the training rows only. Min-max scaling was avoided because the largest spikes would compress most values into a narrow range.
- **Split:** chronological. First 70% for training, next 10% for validation (early stopping and model selection), last 20% for testing.
- **Noise columns:** the two random variables (`rv1`, `rv2`) were dropped.

### Feature engineering

| Group | Features |
|---|---|
| Lagged consumption | 10, 20, 30, 60 and 120 minutes ago, and 24 hours ago, plus the latest change |
| Rolling statistics | mean, standard deviation and maximum over the past 1 hour and 3 hours |
| Time | seconds since midnight, hour of day as sine and cosine, day of week, weekend flag |
| Interaction | indoor-outdoor temperature gap, temperature × humidity, kitchen humidity above the house average, weekend × hour |
| Domain | Belgian public holidays |

Lag periods were chosen from the autocorrelation of consumption: strong short-term dependence (0.75 at 10 minutes, 0.32 at 1 hour) and a daily peak (0.22 at 24 hours). Lags of 6 and 12 hours were left out because their autocorrelation was close to zero. Month was left out because the data covers only 4.5 months, so the test period falls in a month the model barely sees in training.

**Feature selection:** Random Forest importance ranked the 51 features, and subsets of increasing size were compared on the validation set. The top 10 gave the lowest validation error. They are all past consumption or time of day; no temperature or humidity sensor made the top 10.

### Model design

The final architecture is a CNN-LSTM that takes the last 2 hours (12 steps) of the 10 selected features:

```
Conv1D (32 filters, kernel 3, ReLU, causal padding)
  → LSTM (64 units, tanh)
  → Dropout (0.3)
  → Dense (32, ReLU)
  → Dense (1, linear)
```

- **Convolution:** picks up short local patterns, such as the start of a spike. Causal padding stops it from looking ahead within the window.
- **LSTM:** models how those patterns develop over time.
- **Optimizer and loss:** Adam (learning rate 0.001, batch size 64) and mean squared error.
- **Regularisation:** dropout, early stopping (patience 10, best weights restored), and halving the learning rate when validation loss stalls.
- **Final model:** three networks trained with different random seeds, with their predictions averaged.

### Architecture comparison

Each model was trained with three random seeds and scored on the validation set (mean of the three runs):

| Model | Parameters | Validation MAE | Validation RMSE |
|---|---|---|---|
| LSTM | 21,313 | 23.61 | 59.53 |
| GRU | 16,705 | 24.13 | 59.91 |
| **CNN-LSTM** | 27,937 | **23.05** | **58.33** |
| Stacked LSTM | 32,705 | 23.48 | 59.42 |

### Optimisation

- **Random search:** 20 configurations of window length, filters, kernel size, units, dropout, learning rate and batch size. Re-trained with two more seeds, the three best averaged 23.55 to 24.02 validation MAE, worse than the default settings (23.05). Their single-run scores had been partly luck, so the default settings were kept.
- **Dropout experiment:** rates from 0.0 to 0.4, three seeds each. No dropout was clearly worst (24.16); 0.3 was best (23.00).
- **Seed ensemble:** averaging three models reduced validation MAE from 23.00 to 22.55.
- **Before and after optimisation (test set):** MAE 23.70 → 22.80, RMSE 57.55 → 55.71.

## Comparison with published work

| Study | Model | MAE (Wh) | RMSE (Wh) |
|---|---|---|---|
| Candanedo et al. (2017) | Gradient boosting | 35.22 | 66.65 |
| Suranata et al. (2021) | LSTM | 26.98 | 62.01 |
| *Energies* (2023) | Extra Trees | 26.62 | 59.61 |
| **This project** | CNN-LSTM ensemble | **22.80** | **55.71** |

These figures are indicative rather than directly comparable. The earlier studies did not clearly use a chronological split, and they estimated consumption from same-interval sensor readings without past consumption. This project forecasts the next interval under a stricter chronological evaluation.

## Limitations and future work

- Sudden spikes, such as an appliance switching on, often have no warning in the previous readings. The model reacts to them one step late and underestimates their height.
- The model was trained on one house over 4.5 months, so it will not transfer to other homes or seasons without retraining.
- Possible next steps: forecasting further ahead (1 hour or 24 hours), adding occupancy or appliance-level data, and a separate model for the probability of a spike.

## References

1. Candanedo, L. M., Feldheim, V. and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81-97.
2. Suranata, I. W. A., Wardana, I. N. K., Jawas, N. and Aryanto, I. K. A. A. (2021). Feature engineering and long short-term memory for energy use of appliances prediction. *TELKOMNIKA*, 19(3), 920-930.
3. Data-Driven Modeling of Appliance Energy Usage (2023). *Energies*, 16(22), 7536.
4. Kim, T.-Y. and Cho, S.-B. (2019). Predicting residential energy consumption using CNN-LSTM neural networks. *Energy*, 182, 72-81.
5. Runge, J. and Zmeureanu, R. (2021). A review of deep learning techniques for forecasting energy use in buildings. *Energies*, 14(3), 608.