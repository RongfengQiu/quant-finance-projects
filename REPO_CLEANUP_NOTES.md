# Repository Cleanup Notes

## Short diagnosis

The previous reworked version already ran, but it still had several inconsistencies:
- the main ML storyline was mixed with older option-pricing remnants
- the SQLite database was still named `option_pricing.db`
- evaluation did not include a naive historical-volatility baseline
- the cleaned daily date column kept `05:00:00` timestamps because of unnecessary UTC conversion
- the main EDA notebook did not clearly match the current ML volatility-forecasting project
- `__pycache__` and compiled Python files were still present in the repository

## Files modified

- `README.md`
- `.gitignore`
- `pipeline.py`
- `requirements.txt`
- `sql/evaluation_queries.sql`
- `dashboard/streamlit_app.py`
- `src/config.py`
- `src/data/load_and_clean_data.py`
- `src/models/train_models.py`
- `src/evaluation/evaluate_models.py`
- `notebooks/eda.ipynb`

## Files moved to legacy

- `src/models/black_scholes.py` → `legacy/option_pricing/black_scholes.py`
- `src/models/monte_carlo.py` → `legacy/option_pricing/monte_carlo.py`
- `src/models/ml_baseline.py` → `legacy/option_pricing/ml_baseline.py`
- `src/features/option_dataset.py` → `legacy/option_pricing/option_dataset.py`
- `src/volatility/estimate_volatility.py` → `legacy/option_pricing/estimate_volatility.py`
- `src/evaluation/visualization.py` → `legacy/option_pricing/visualization.py`
- `report/` → `legacy/option_pricing/old_report_assets/`

## Revised repository story

Main project:
- predict future 20-day annualized realized volatility
- use a chronological supervised ML workflow
- compare ML models with a naive EWMA volatility baseline

Legacy content:
- older option-pricing modules are archived under `legacy/option_pricing/`
- they are not part of the current main pipeline

## Final verification

Verified after cleanup:
- main pipeline runs successfully
- database is now `database/vol_forecasting.db`
- validation metrics include the naive baseline
- test metrics include the naive baseline
- cleaned date column is now a natural daily timestamp without `05:00:00`
- `__pycache__` and `.pyc` files were removed
