from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import CONFIG
from pipeline import run_pipeline


st.set_page_config(page_title="Option Pricing DS Workflow", layout="wide")

st.title("Option Pricing – Data Science Workflow")
st.caption("End-to-end pipeline: data → features → split → pricing → evaluation → SQL")

with st.sidebar:
    st.header("Run pipeline")
    recompute = st.checkbox("Recompute from scratch", value=False)
    n_sim = st.number_input("Monte Carlo simulations", min_value=1000, max_value=200000, value=CONFIG.n_sim, step=1000)
    sample_every = st.number_input("Sample every N days", min_value=1, max_value=200, value=CONFIG.sample_every_n_days, step=1)

    if st.button("Run / Update results"):
        run_pipeline(recompute=recompute, n_sim=int(n_sim), sample_every_n_days=int(sample_every))
        st.success("Pipeline finished.")

results_path = Path("data/results/pricing_results.csv")
agg_path = Path("data/results/error_by_T.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pricing results (sample)")
    if results_path.exists():
        df = pd.read_csv(results_path)
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("Run the pipeline to generate data/results/pricing_results.csv")

with col2:
    st.subheader("Error summary by maturity")
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        st.dataframe(agg, use_container_width=True)

        fig = plt.figure()
        plt.plot(agg["T"], agg["mean_abs_error"], marker="o")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Mean Absolute Error")
        plt.title("MC vs BS – MAE by maturity")
        st.pyplot(fig)
    else:
        st.info("Run the pipeline to generate data/results/error_by_T.csv")

st.subheader("Saved figures")
fig_dir = Path("reports/figures")
if fig_dir.exists():
    figs = sorted(fig_dir.glob("*.png"))
    if figs:
        for p in figs:
            st.image(str(p), caption=p.name)
    else:
        st.caption("No figures yet. Run pipeline to generate plots.")
