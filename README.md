# Appliance Energy Prediction

Multivariate time-series forecasting of household appliance energy use with deep learning.

The goal is to predict `Appliances` (energy use in Wh) for the next 10-minute interval, using only information available before that interval.

**Status:** in progress. Data analysis, preprocessing, feature engineering and baseline models are done. The deep learning model, optimisation and final report are still to come.

## Dataset

[Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) from the UCI Machine Learning Repository (CC BY 4.0).

- 19,735 readings at 10-minute intervals, from 11 January to 27 May 2016
- 29 columns: appliance and lighting energy use, temperature and humidity in nine areas of a house, and outdoor weather
- No missing values, no duplicate timestamps and no gaps in the timeline

The file is stored at `data/raw/energy_data_set.csv`.

## Project structure

```
├── data/
│   ├── raw/                  # original dataset
│   └── processed/            # engineered feature table
├── notebooks/
│   └── EDA.ipynb             # analysis, features, baselines
├── src/
│   ├── data_preprocessing.py # loading, chronological split, scaling
│   └── feature_engineering.py# lag, rolling, time and interaction features
├── models/                   # trained models
├── reports/                  # final report
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11.

```
git clone <repository-url>
cd energy-prediction
python -m venv venv
venv\Scripts\activate          # Mac or Linux: source venv/bin/activate
pip install -r requirements.txt
```

## How to run

1. Start Jupyter with `jupyter notebook`, or open the folder in VS Code and select the `venv` kernel.
2. Open `notebooks/EDA.ipynb`.
3. Run all cells from top to bottom.

The notebook loads the raw data, builds the features with `src/feature_engineering.py`, saves them to `data/processed/features.csv`, and trains the baseline models.

## Approach

**Preventing data leakage**

- The data is split in time order: first 70% for training, next 10% for validation, last 20% for testing.
- Every sensor reading and every consumption-based feature is shifted back by at least one step, so no feature uses information from the interval being predicted.
- The scaler is fitted on the training rows only.

**Preprocessing**

- Outliers are kept. About 10.8% of readings are above the IQR upper bound of 175 Wh, which is too many to be sensor errors; they are real periods of high use.
- The target is trained as `log(1 + Appliances)` to reduce its heavy right skew. All metrics are reported in Wh.
- Features are standardised. Min-max scaling was avoided because the largest spikes would compress most values into a narrow range.
- The two random noise columns (`rv1`, `rv2`) are dropped.

**Feature engineering**

| Group | Features |
|---|---|
| Lagged consumption | 10, 20, 30, 60 and 120 minutes ago, and 24 hours ago, chosen from the autocorrelation analysis |
| Rolling statistics | mean, standard deviation and maximum over the past 1 hour and 3 hours |
| Time | seconds since midnight, hour of day as sine and cosine, day of week (0 = Monday), weekend flag |
| Interaction | indoor-outdoor temperature gap, temperature × humidity, kitchen humidity above the house average, weekend × hour |
| Domain | Belgian public holidays |

Month is left out on purpose: the data covers only 4.5 months, so the test period falls in a month the model barely sees in training.

## Results so far

Baseline models on the test set (30 April to 27 May 2016):

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Naive (last value) | 26.05 | 65.19 | 21.86 | 0.45 |
| Linear Regression | 27.26 | 66.49 | 22.12 | 0.43 |
| Random Forest | 26.51 | 58.30 | 23.19 | 0.56 |

Repeating the last reading is a strong baseline at a 10-minute horizon. Neither model beats it on MAE; the Random Forest improves RMSE by about 10%.

## Reference

Candanedo, L. M., Feldheim, V. and Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81-97.