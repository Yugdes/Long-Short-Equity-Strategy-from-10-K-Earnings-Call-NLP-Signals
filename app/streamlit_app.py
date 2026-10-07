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
        
        # Marketing metrics if available
        has_marketing = "market_orientation" in panel.columns
        agg_dict = {
            "op_risk_pct": "mean",
            "non_op_risk_pct": "mean"
        }
        if has_marketing:
            agg_dict["market_orientation"] = "mean"
            agg_dict["marketing_capabilities"] = "mean"
            
        # Get industry from ffi48 or sic
        # The Astvansh dataset might have ffi48 or we can just group by all stocks as one industry
        panel["industry"] = "All"
        
        # Aggregate to yearly for the app
        df = panel.groupby(["ticker", "year", "industry"]).agg(agg_dict).reset_index()
        
        # Industry medians
        ind_med = df.groupby(["industry", "year"])["op_risk_pct"].median().reset_index()
        ind_med = ind_med.rename(columns={"op_risk_pct": "industry_median_pct"})
        
        df = df.merge(ind_med, on=["industry", "year"], how="left")
        
        # Filter to tickers with enough data
        counts = df["ticker"].value_counts()
        valid_tickers = counts[counts >= 5].index
        df = df[df["ticker"].isin(valid_tickers)]
        
        # Load ML predictions if available
        ml_file = Path("data/processed/ml_predictions.parquet")
        if ml_file.exists():
            ml_df = pd.read_parquet(ml_file)
            if "year" not in ml_df.columns:
                ml_df["year"] = ml_df["month_start"].dt.year if "month_start" in ml_df.columns else ml_df["month"].dt.year
            ml_agg = ml_df.groupby(["ticker", "year"])["ml_prediction"].mean().reset_index()
            df = df.merge(ml_agg, on=["ticker", "year"], how="left")
            
        return df, has_marketing
    except Exception as e:
        st.error(f"Failed to load real data: {e}")
        return pd.DataFrame(), False

# --- Main App ---
st.title("📊 Operational Risk Benchmark")
st.markdown("""
*Based on 10-K NLP risk disclosures. Research and educational use only.*
""")

try:
    df, has_marketing = load_data()
    
    # Sidebar
    st.sidebar.header("Configuration")
    selected_ticker = st.sidebar.selectbox("Select Company (Ticker)", df["ticker"].unique())
    
    # Filter data
    company_data = df[df["ticker"] == selected_ticker].sort_values("year")
    latest_year = company_data["year"].max()
    latest_data = company_data[company_data["year"] == latest_year].iloc[0]
    
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    op_risk = latest_data["op_risk_pct"]
    prev_op_risk = company_data[company_data["year"] == latest_year - 1]["op_risk_pct"].iloc[0] if len(company_data) > 1 else op_risk
    delta = op_risk - prev_op_risk
    
    col1.metric("Operational Risk %ile", f"{op_risk:.1f}", f"{delta:.1f} YoY")
    col2.metric("Non-Op Risk %ile", f"{latest_data['non_op_risk_pct']:.1f}")
    
    if has_marketing and "market_orientation" in latest_data:
        mkt_score = latest_data["market_orientation"]
        prev_mkt = company_data[company_data["year"] == latest_year - 1]["market_orientation"].iloc[0] if len(company_data) > 1 else mkt_score
        mkt_delta = mkt_score - prev_mkt
        col3.metric("Mktg Orientation Score", f"{mkt_score:.2f}", f"{mkt_delta:.2f} YoY")
    else:
        col3.metric("Mktg Orientation Score", "N/A")
        
    col4.metric("Industry", latest_data["industry"])
    
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

    if has_marketing and "market_orientation" in company_data.columns:
        st.subheader("Marketing Emphasis Over Time")
        fig_mkt = px.line(company_data, x="year", y=["market_orientation", "marketing_capabilities"],
                          title=f"{selected_ticker} Marketing Strategy Scores")
        st.plotly_chart(fig_mkt, use_container_width=True)
        
    if "ml_prediction" in company_data.columns:
        st.subheader("Machine Learning: Expected Excess Return Rank")
        st.markdown("Higher rank implies the ML model expects higher excess returns relative to peers next month.")
        fig_ml = px.bar(company_data.dropna(subset=["ml_prediction"]), x="year", y="ml_prediction",
                        title=f"{selected_ticker} ML Expected Rank (Z-Score/Normalized)")
        st.plotly_chart(fig_ml, use_container_width=True)
        
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Run the data pipeline first to generate the required parquet files.")
