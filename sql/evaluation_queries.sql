-- 1. Validation performance of all models, including the naive volatility baseline
SELECT *
FROM validation_metrics
ORDER BY rmse ASC, mae ASC;

-- 2. Test performance of all models after validation-based model selection
SELECT *
FROM test_metrics
ORDER BY rmse ASC, mae ASC;

-- 3. Selected features saved by the pipeline
SELECT *
FROM feature_target_correlation
ORDER BY abs_corr_with_target DESC;

-- 4. Largest prediction errors for the selected final model on the test set
SELECT date,
       model_name,
       future_20d_rv_ann,
       prediction,
       residual
FROM test_predictions
WHERE is_selected_model = 1
ORDER BY ABS(residual) DESC
LIMIT 20;

-- 5. Monthly average actual vs predicted volatility for the selected final model
SELECT SUBSTR(date, 1, 7) AS year_month,
       AVG(future_20d_rv_ann) AS avg_actual_vol,
       AVG(prediction) AS avg_predicted_vol,
       COUNT(*) AS n_rows
FROM test_predictions
WHERE is_selected_model = 1
GROUP BY SUBSTR(date, 1, 7)
ORDER BY year_month;
