from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pipeline import run_pipeline


st.set_page_config(page_title="Financial ML Workflow", layout="wide")

st.title("Future Realized Volatility Forecasting")
st.caption(
    "Student-level end-to-end ML workflow using historical stock-price features and a naive volatility baseline"
)

with st.sidebar:
    st.header("Run pipeline")
    recompute = st.checkbox("Recompute from scratch", value=False)
    if st.button("Run / Update results"):
        run_pipeline(recompute=recompute)
        st.success("Pipeline finished.")

val_metrics_path = Path("data/results/validation_metrics.csv")
test_metrics_path = Path("data/results/test_metrics.csv")
predictions_path = Path("data/results/test_predictions.csv")
selected_path = Path("data/results/selected_features.csv")
fig_dir = Path("reports/figures")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Validation metrics")
    if val_metrics_path.exists():
        st.dataframe(pd.read_csv(val_metrics_path), width='stretch')
    else:
        st.info("Run the pipeline first to generate validation metrics.")

with col2:
    st.subheader("Test metrics")
    if test_metrics_path.exists():
        st.dataframe(pd.read_csv(test_metrics_path), width='stretch')
    else:
        st.info("Run the pipeline first to generate test metrics.")

st.subheader("Selected features")
if selected_path.exists():
    st.dataframe(pd.read_csv(selected_path), width='stretch')
else:
    st.info("Run the pipeline first to generate the selected feature list.")

st.subheader("Selected-model predictions (sample)")
if predictions_path.exists():
    pred = pd.read_csv(predictions_path)
    selected_pred = pred.loc[pred["is_selected_model"] == 1].copy()
    st.dataframe(selected_pred.head(50), width='stretch')

    fig = plt.figure(figsize=(7, 4))
    plt.plot(selected_pred["future_20d_rv_ann"].head(100).values, label="Actual")
    plt.plot(selected_pred["prediction"].head(100).values, label="Predicted")
    model_name = selected_pred["model_name"].iloc[0] if not selected_pred.empty else "selected model"
    plt.title(f"Test sample: actual vs predicted ({model_name})")
    plt.legend()
    st.pyplot(fig)
else:
    st.info("Run the pipeline first to generate predictions.")

st.subheader("Saved figures")
if fig_dir.exists():
    figs = sorted(fig_dir.glob("*.png"))
    if figs:
        for fig_path in figs:
            st.image(str(fig_path), caption=fig_path.name)
    else:
        st.caption("No figures generated yet.")
