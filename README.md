# Future Realized Volatility Forecasting – End-to-End Machine Learning Workflow

**Author:** Rongfeng Qiu  
**Student ID:** 488004

## 1. Project Goal

This repository is a portfolio-style **end-to-end machine learning project** with a financial context.

**Main objective:** use historical stock-price-based features to predict the **next 20 trading days of annualized realized volatility**.

The project is intentionally kept at a realistic master's-student level:
- complete workflow
- interpretable features
- chronological train/validation/test split
- train-only feature screening
- simple baseline models
- honest comparison with a naive volatility baseline
- clear evaluation and presentation

## 2. Main ML Workflow

The main pipeline follows this storyline:

raw stock prices  
→ cleaning  
→ feature engineering  
→ supervised ML dataset  
→ chronological split  
→ train-only feature screening  
→ model training  
→ validation-based model selection  
→ final test evaluation  
→ result storage  
→ dashboard / presentation

## 3. Target Variable

The target is:

```text
future_20d_rv_ann
```

It is defined as the **annualized realized volatility computed from the next 20 trading days' log returns**.

For each date `t`:
- the features use information available up to time `t`
- the target uses log returns from `t+1` to `t+20`

This avoids look-ahead bias and makes the project a natural financial ML task.

## 4. Core Feature Set

The first version of the project uses a small, interpretable feature set:

- `log_return`
- `ret_5d`
- `ret_20d`
- `rv_5d_ann`
- `rv_20d_ann`
- `rv_60d_ann`
- `ewma_vol_ann`
- `ma_20_ratio`
- `ma_60_ratio`
- `month`

These features are based on:
- historical returns
- historical volatility
- simple trend information
- a light calendar effect

## 5. Models and Baseline

The project compares four predictors:

- **Linear Regression**
- **Ridge Regression**
- **Random Forest Regressor**
- **NaiveEWMA baseline** (`prediction = current ewma_vol_ann`)

The best predictor is selected using the **validation set**, and all predictors are then reported again on the **test set**.

This comparison is important because **simple historical-volatility proxies are often strong baselines in financial forecasting**. The purpose of the project is not to claim that ML always beats those baselines. Instead, the goal is to demonstrate a realistic and properly structured time-series ML workflow.

## 6. Repository Structure

```text
.
├── pipeline.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_and_clean_data.py
│   │   └── dirty_data_cleaning_demo.ipynb
│   ├── features/
│   │   ├── feature_engineering.py
│   │   ├── make_ml_dataset.py
│   │   └── feature_selection.py
│   ├── split/
│   │   └── time_split.py
│   ├── models/
│   │   └── train_models.py
│   ├── evaluation/
│   │   ├── evaluate_models.py
│   │   └── metrics.py
│   └── database/
│       └── db_utils.py
├── legacy/
│   └── option_pricing/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── results/
├── database/
├── sql/
├── reports/figures/
├── notebooks/
│   └── eda.ipynb
└── dashboard/
```

## 7. Dataset Notes

The repository contains two CSV files with different purposes:

- `AAPL_historical_data.csv`  
  The main raw dataset used in the full ML workflow.

- `AAPL_dirty_practice_comma.csv`  
  A separate intentionally messy practice dataset used to demonstrate data-cleaning and data-repair ability.

The dirty practice dataset is **not** used in the main ML results.
It is kept as a separate skill-demonstration file.

## 8. Legacy Modules

Earlier option-pricing files are still included in the repository, but they are now stored under:

```text
legacy/option_pricing/
```

They are kept only as archived finance extensions and are **not part of the current main pipeline**.
They are preserved for reference only and are not required for the active ML workflow.

## 9. How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the main pipeline from the project root:

```bash
python pipeline.py --recompute
```

Launch the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

## 10. Main Outputs

After running the pipeline, the main generated files are:

### Processed data
- `data/processed/cleaned_prices.csv`
- `data/processed/features_prices.csv`
- `data/processed/ml_dataset.csv`

### Chronological splits
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

### Feature-screening reports
- `data/results/feature_variance_report.csv`
- `data/results/feature_target_correlation.csv`
- `data/results/feature_high_correlation_pairs.csv`
- `data/results/selected_features.csv`

### Model results
- `data/results/validation_metrics.csv`
- `data/results/test_metrics.csv`
- `data/results/test_predictions.csv`

### Figures
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/residual_distribution.png`
- `reports/figures/feature_importance.png` *(only when the selected model exposes feature importances)*

### Database
- `database/vol_forecasting.db`

## 11. Notes for Recruiters / Reviewers

This project is designed to show:
- a full machine-learning workflow
- awareness of time-series leakage risk
- simple but meaningful financial feature engineering
- validation-based model selection
- honest comparison against a naive volatility baseline
- basic result storage and presentation

The project should be read primarily as a **student-level ML finance workflow project**, not as an attempt to claim a production-grade alpha signal.
