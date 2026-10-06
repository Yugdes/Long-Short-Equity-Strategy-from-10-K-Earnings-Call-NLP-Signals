"""
Streamlit App — Operational Risk Benchmark.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os

# --- Page Config ---
st.set_page_config(
    page_title="Operational Risk Benchmark",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load the processed small parquet files for the app."""
    try:
        panel = pd.read_parquet("data/processed/panel_stock_month.parquet")
        
        # We need ticker, year, op_risk_pct, non_op_risk_pct, industry
        if "ticker" not in panel.columns:
            sec = pd.read_parquet("data/processed/security_master.parquet")
            panel = panel.merge(sec[["cik", "ticker"]].drop_duplicates("cik"), on="cik", how="left")
            
        # Convert month to year
        panel["year"] = panel["month"].dt.year if hasattr(panel["month"].dt, "year") else panel["month"].apply(lambda x: x.year)
        
        # Convert risk signals to percentiles globally per year
        panel["op_risk_pct"] = panel.groupby("year")["S1_OpRisk"].rank(pct=True) * 100
        panel["non_op_risk_pct"] = panel.groupby("year")["S2_NonOpRisk"].rank(pct=True) * 100
        
        # Get industry from ffi48 or sic
        # The Astvansh dataset might have ffi48 or we can just group by all stocks as one industry
        panel["industry"] = "All"
        
        # Aggregate to yearly for the app
        df = panel.groupby(["ticker", "year", "industry"]).agg({
            "op_risk_pct": "mean",
            "non_op_risk_pct": "mean"
        }).reset_index()
        
        # Industry medians
        ind_med = df.groupby(["industry", "year"])["op_risk_pct"].median().reset_index()
        ind_med = ind_med.rename(columns={"op_risk_pct": "industry_median_pct"})
        
        df = df.merge(ind_med, on=["industry", "year"], how="left")
        
        # Filter to tickers with enough data
        counts = df["ticker"].value_counts()
        valid_tickers = counts[counts >= 5].index
        df = df[df["ticker"].isin(valid_tickers)]
        
        return df
    except Exception as e:
        st.error(f"Failed to load real data: {e}")
        return pd.DataFrame()

# --- Main App ---
st.title("📊 Operational Risk Benchmark")
st.markdown("""
*Based on 10-K NLP risk disclosures. Research and educational use only.*
""")

try:
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Configuration")
    selected_ticker = st.sidebar.selectbox("Select Company (Ticker)", df["ticker"].unique())
    
    # Filter data
    company_data = df[df["ticker"] == selected_ticker].sort_values("year")
    latest_year = company_data["year"].max()
    latest_data = company_data[company_data["year"] == latest_year].iloc[0]
    
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    
    op_risk = latest_data["op_risk_pct"]
    prev_op_risk = company_data[company_data["year"] == latest_year - 1]["op_risk_pct"].iloc[0] if len(company_data) > 1 else op_risk
    delta = op_risk - prev_op_risk
    
    col1.metric("Operational Risk Percentile", f"{op_risk:.1f}", f"{delta:.1f} YoY")
    col2.metric("Non-Operational Risk Percentile", f"{latest_data['non_op_risk_pct']:.1f}")
    col3.metric("Industry", latest_data["industry"])
    
    # Time Series Chart
    st.subheader("Risk Evolution vs Industry Peer Median")
    
    fig = px.line(company_data, x="year", y=["op_risk_pct", "industry_median_pct"], 
                  labels={"value": "Percentile", "variable": "Metric"},
                  title=f"{selected_ticker} Operational Risk Trajectory")
    
    # Customize names
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("op_risk_pct", selected_ticker)
                                                 .replace("industry_median_pct", "Industry Median")))
                                                 
    fig.update_layout(yaxis_range=[0, 100], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Run the data pipeline first to generate the required parquet files.")
