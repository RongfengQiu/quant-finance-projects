from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG


@dataclass
class FeatureScreeningResult:
    selected_features: list[str]
    variance_table: pd.DataFrame
    target_correlation_table: pd.DataFrame
    high_corr_pairs_table: pd.DataFrame


def light_feature_screening(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> FeatureScreeningResult:
    """Apply light screening using training data only."""

    work = train_df[feature_cols + [target_col]].copy()

    variance_rows = []
    current_features = []
    for col in feature_cols:
        var = float(work[col].var())
        near_zero = var <= CONFIG.variance_threshold
        variance_rows.append({"feature": col, "variance": var, "near_zero_variance": near_zero})
        if not near_zero:
            current_features.append(col)

    variance_table = pd.DataFrame(variance_rows).sort_values("feature").reset_index(drop=True)

    target_corr_rows = []
    for col in current_features:
        corr = float(work[col].corr(work[target_col]))
        target_corr_rows.append({"feature": col, "corr_with_target": corr, "abs_corr_with_target": abs(corr)})

    target_correlation_table = pd.DataFrame(target_corr_rows).sort_values(
        "abs_corr_with_target", ascending=False
    ).reset_index(drop=True)

    target_corr_map = dict(
        zip(target_correlation_table["feature"], target_correlation_table["abs_corr_with_target"])
    )

    corr_matrix = work[current_features].corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask)

    high_corr_rows = []
    to_drop: set[str] = set()
    for row_feature in upper.index:
        for col_feature in upper.columns:
            corr_value = upper.loc[row_feature, col_feature]
            if pd.isna(corr_value) or corr_value < CONFIG.correlation_threshold:
                continue

            row_score = target_corr_map.get(row_feature, 0.0)
            col_score = target_corr_map.get(col_feature, 0.0)
            drop_feature = row_feature if row_score < col_score else col_feature

            high_corr_rows.append(
                {
                    "feature_1": row_feature,
                    "feature_2": col_feature,
                    "abs_corr": float(corr_value),
                    "drop_recommendation": drop_feature,
                }
            )
            to_drop.add(drop_feature)

    high_corr_pairs_table = pd.DataFrame(high_corr_rows)
    if not high_corr_pairs_table.empty:
        high_corr_pairs_table = high_corr_pairs_table.sort_values(
            ["abs_corr", "feature_1", "feature_2"], ascending=[False, True, True]
        ).reset_index(drop=True)

    selected_features = [col for col in current_features if col not in to_drop]

    return FeatureScreeningResult(
        selected_features=selected_features,
        variance_table=variance_table,
        target_correlation_table=target_correlation_table,
        high_corr_pairs_table=high_corr_pairs_table,
    )
