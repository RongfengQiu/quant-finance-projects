-- Option Pricing Project: SQL Evaluation Examples
-- Demonstrates JOIN + FILTER + AGGREGATION on SQLite.

-- 1) JOIN (options with features) + FILTER (only TEST split, maturities >= 0.5)
SELECT
  o.date,
  o.split,
  o.S,
  o.K,
  o.T,
  o.r,
  o.sigma_used,
  o.bs_price,
  o.mc_price,
  o.abs_error,
  f.rv_30d_ann,
  f.ewma_vol_ann
FROM options_dataset o
LEFT JOIN features f
  ON f.date = o.date
WHERE o.split = 'test'
  AND o.T >= 0.5
ORDER BY o.date, o.T, o.K;

-- 2) AGGREGATION: MAE by maturity (TEST split)
SELECT
  T,
  AVG(abs_error) AS mean_abs_error,
  COUNT(*) AS n
FROM options_dataset
WHERE split = 'test'
GROUP BY T
ORDER BY T;

-- 3) AGGREGATION: MAE by year (TEST split)
SELECT
  substr(date, 1, 4) AS year,
  AVG(abs_error) AS mean_abs_error,
  COUNT(*) AS n
FROM options_dataset
WHERE split = 'test'
GROUP BY substr(date, 1, 4)
ORDER BY year;

-- 4) AGGREGATION: MAE by moneyness bucket (K/S)
SELECT
  CASE
    WHEN (K / S) < 0.9 THEN 'deep_ITM'
    WHEN (K / S) < 1.0 THEN 'ITM'
    WHEN (K / S) < 1.1 THEN 'OTM'
    ELSE 'deep_OTM'
  END AS moneyness_bucket,
  AVG(abs_error) AS mean_abs_error,
  COUNT(*) AS n
FROM options_dataset
WHERE split = 'test'
GROUP BY moneyness_bucket
ORDER BY mean_abs_error DESC;
