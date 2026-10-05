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
    # In a real app, this would load from a data/processed/app_data/ folder.
    # For now, we simulate the data structure to demonstrate the UI.
    
    # Dummy data for demonstration
    np.random.seed(42)
    years = list(range(2010, 2025))
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "WMT", "TSLA"]
    
    data = []
    for t in tickers:
        base_risk = np.random.uniform(0.1, 0.8)
        for y in years:
            risk = np.clip(base_risk + np.random.normal(0, 0.05), 0, 1)
            data.append({
                "ticker": t,
                "year": y,
                "op_risk_pct": risk * 100,
                "non_op_risk_pct": np.random.uniform(20, 80),
                "industry": "Tech" if t in ["AAPL", "MSFT"] else "Other",
                "ff48": 35 if t in ["AAPL", "MSFT"] else 48
            })
            
    df = pd.DataFrame(data)
    
    # Industry medians
    ind_med = df.groupby(["industry", "year"])["op_risk_pct"].median().reset_index()
    ind_med = ind_med.rename(columns={"op_risk_pct": "industry_median_pct"})
    
    df = df.merge(ind_med, on=["industry", "year"], how="left")
    
    return df

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
