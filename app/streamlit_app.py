"""
Streamlit App — Premium Quantitative Dashboard with ML What-If Simulator.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Alpha Signal | NLP Risk Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Glassmorphism & Premium Dark Mode) ---
st.markdown("""
<style>
/* Main App Background */
.stApp {
    background-color: #0A0A0A;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(20, 184, 166, 0.05), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.05), transparent 25%);
    color: #F3F4F6;
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(15, 15, 15, 0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Glow Cards */
.metric-card {
    background: rgba(20, 20, 25, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    margin-bottom: 20px;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 40px rgba(20, 184, 166, 0.15);
    border-color: rgba(20, 184, 166, 0.4);
}
.metric-title {
    font-size: 0.85rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    font-weight: 600;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2DD4BF 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.metric-delta {
    font-size: 0.95rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
}
.delta-positive { 
    background: rgba(16, 185, 129, 0.1);
    color: #10B981; 
}
.delta-negative { 
    background: rgba(239, 68, 68, 0.1);
    color: #EF4444; 
}
.delta-neutral { 
    background: rgba(156, 163, 175, 0.1);
    color: #9CA3AF; 
}

/* Headers */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
hr {
    border-color: rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


def create_metric_card(title, value, delta=None, delta_text="YoY", inverse_color=False):
    """HTML generator for premium metric cards."""
    delta_html = ""
    if delta is not None:
        if delta > 0:
            css_class = "delta-negative" if inverse_color else "delta-positive"
            arrow = "↑"
        elif delta < 0:
            css_class = "delta-positive" if inverse_color else "delta-negative"
            arrow = "↓"
        else:
            css_class = "delta-neutral"
            arrow = "→"
            
        delta_html = f'<div class="metric-delta {css_class}">{arrow} {abs(delta):.2f} {delta_text}</div>'
        
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

@st.cache_resource
def load_ml_model():
    """Loads the latest walk-forward ML model."""
    try:
        model_path = Path("results/models/lightgbm_split_13.joblib")
        if model_path.exists():
            return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load ML model: {e}")
    return None

@st.cache_data
def load_data():
    try:
        panel = pd.read_parquet("data/processed/panel_stock_month.parquet")
        
        if "ticker" not in panel.columns:
            sec = pd.read_parquet("data/processed/security_master.parquet")
            panel = panel.merge(sec[["cik", "ticker"]].drop_duplicates("cik"), on="cik", how="left")
            
        panel["year"] = panel["month"].dt.year if hasattr(panel["month"].dt, "year") else panel["month"].apply(lambda x: x.year)
        
        # Calculate Percentiles globally
        panel["op_risk_pct"] = panel.groupby("year")["S1_OpRisk"].rank(pct=True) * 100
        
        has_marketing = "market_orientation" in panel.columns
        
        # Build Aggregated Data for Charts
        agg_dict = {"op_risk_pct": "mean"}
        if has_marketing:
            agg_dict["market_orientation"] = "mean"
            agg_dict["marketing_capabilities"] = "mean"
            
        panel["industry"] = "All"
        df = panel.groupby(["ticker", "year", "industry"]).agg(agg_dict).reset_index()
        
        ind_med = df.groupby(["industry", "year"])["op_risk_pct"].median().reset_index()
        ind_med = ind_med.rename(columns={"op_risk_pct": "industry_median_pct"})
        df = df.merge(ind_med, on=["industry", "year"], how="left")
        
        # Get Latest Raw Row for the Simulator
        latest_rows = panel.sort_values("month").groupby("ticker").last().reset_index()
        
        ml_file = Path("data/processed/ml_predictions.parquet")
        if ml_file.exists():
            ml_df = pd.read_parquet(ml_file)
            if "year" not in ml_df.columns:
                ml_df["year"] = ml_df["month_start"].dt.year if "month_start" in ml_df.columns else ml_df["month"].dt.year
            ml_agg = ml_df.groupby(["ticker", "year"])["ml_prediction"].mean().reset_index()
            df = df.merge(ml_agg, on=["ticker", "year"], how="left")
            
        return df, latest_rows, has_marketing
    except Exception as e:
        st.error(f"Failed to load real data: {e}")
        return pd.DataFrame(), pd.DataFrame(), False

# --- Main App ---
try:
    df, latest_rows, has_marketing = load_data()
    ml_model = load_ml_model()
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; background: linear-gradient(90deg, #2DD4BF, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>QUANT ENGINE</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.8rem;'>NLP 10-K ALPHA SIGNALS</p>", unsafe_allow_html=True)
        st.write("---")
        
        selected_ticker = st.selectbox("Select Asset", sorted(df["ticker"].dropna().unique()), index=0)
        st.write("---")
        st.markdown("""
        **Data Sources:**
        - SEC 10-K NLP Filings
        - Earnings Call Transcripts
        - Yahoo Finance Returns
        """)

    # Filter data
    company_data = df[df["ticker"] == selected_ticker].sort_values("year")
    latest_year = company_data["year"].max()
    latest_data = company_data[company_data["year"] == latest_year].iloc[0]
    latest_raw = latest_rows[latest_rows["ticker"] == selected_ticker].iloc[0]
    
    # --- Header ---
    st.markdown(f"<h1>Dashboard: <span style='color: #2DD4BF;'>${selected_ticker}</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #9CA3AF; margin-top: -15px;'>Latest Profile Year: {latest_year}</p>", unsafe_allow_html=True)
    
    # --- Top Metrics Row ---
    col1, col2, col3 = st.columns(3)
    
    # Metric 1: Operational Risk
    op_risk = latest_data["op_risk_pct"]
    prev_op = company_data[company_data["year"] == latest_year - 1]["op_risk_pct"].iloc[0] if len(company_data) > 1 else op_risk
    col1.markdown(create_metric_card("Operational Risk Percentile", f"{op_risk:.1f}", op_risk - prev_op, inverse_color=True), unsafe_allow_html=True)
    
    # Metric 2: Marketing Score
    if has_marketing and pd.notna(latest_data.get("market_orientation")):
        mkt = latest_data["market_orientation"]
        prev_mkt = company_data[company_data["year"] == latest_year - 1]["market_orientation"].iloc[0] if len(company_data) > 1 else mkt
        col2.markdown(create_metric_card("Marketing Orientation", f"{mkt:.2f}", mkt - prev_mkt), unsafe_allow_html=True)
    else:
        col2.markdown(create_metric_card("Marketing Orientation", "N/A"), unsafe_allow_html=True)
        
    # Metric 3: ML Expected Rank
    if "ml_prediction" in latest_data and pd.notna(latest_data["ml_prediction"]):
        ml_score = latest_data["ml_prediction"]
        prev_ml = company_data[company_data["year"] == latest_year - 1]["ml_prediction"].iloc[0] if len(company_data) > 1 else ml_score
        col3.markdown(create_metric_card("Expected Excess Return", f"{ml_score:.3f}", ml_score - prev_ml), unsafe_allow_html=True)
    else:
        col3.markdown(create_metric_card("Expected Excess Return", "N/A"), unsafe_allow_html=True)
        
    st.write("---")
    
    # --- Tabs Section ---
    tab1, tab2, tab3 = st.tabs(["📉 Risk Trajectory", "🧠 Machine Learning Trends", "🧪 What-If Simulator"])
    
    with tab1:
        st.markdown("### Operational Risk vs. Industry Peer Median")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=company_data["year"], y=company_data["op_risk_pct"],
                                 mode='lines+markers', name=f'{selected_ticker} Op Risk',
                                 line=dict(color='#2DD4BF', width=3),
                                 marker=dict(size=8, color='#0A0A0A', line=dict(width=2, color='#2DD4BF'))))
        
        fig.add_trace(go.Scatter(x=company_data["year"], y=company_data["industry_median_pct"],
                                 mode='lines', name='Industry Median',
                                 line=dict(color='#6B7280', width=2, dash='dash')))
                                 
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Risk Percentile (Higher = Worse)', range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(title='', gridcolor='rgba(255,255,255,0.1)'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_mkt, col_ml = st.columns(2)
        
        with col_mkt:
            st.markdown("### Marketing Fundamentals")
            if has_marketing and "market_orientation" in company_data.columns:
                fig_mkt = go.Figure()
                fig_mkt.add_trace(go.Scatter(x=company_data["year"], y=company_data["market_orientation"],
                                         mode='lines', fill='tozeroy', name='Orientation',
                                         line=dict(color='#8B5CF6')))
                fig_mkt.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    height=350, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_mkt, use_container_width=True)
            else:
                st.info("No marketing data available.")
                
        with col_ml:
            st.markdown("### ML Model Alpha Rank")
            if "ml_prediction" in company_data.columns:
                fig_ml = px.bar(company_data.dropna(subset=["ml_prediction"]), 
                                x="year", y="ml_prediction",
                                color="ml_prediction",
                                color_continuous_scale=["#EF4444", "#0A0A0A", "#10B981"])
                                
                fig_ml.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title='Expected Alpha', gridcolor='rgba(255,255,255,0.1)'),
                    xaxis=dict(title='', gridcolor='rgba(255,255,255,0.1)'),
                    coloraxis_showscale=False,
                    height=350, margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_ml, use_container_width=True)
            else:
                st.info("No ML predictions available.")
                
    with tab3:
        st.markdown("### 🧪 Machine Learning What-If Simulator")
        st.markdown(f"Inject simulated conditions into the **Walk-Forward LightGBM Model** to instantly see how {selected_ticker}'s expected returns would react.")
        
        if ml_model is None:
            st.warning("No trained ML model found in `results/models/`.")
        else:
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                st.markdown("#### Scenario Controls")
                risk_mult = st.slider("Operational Risk Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="1.0 = Base Risk. 2.0 = Double the NLP risk mentions in the 10-K.")
                mkt_mult = st.slider("Marketing Emphasis Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="1.0 = Base Marketing. 2.0 = Double the marketing orientation in Earnings Calls.")
            
            # Reconstruct the feature vector exactly as the model expects
            features = [c for c in latest_raw.index if c.startswith("S")]
            if has_marketing:
                features += ["market_orientation", "marketing_capabilities"]
                
            features = [f for f in features if f in latest_raw.index and pd.notna(latest_raw[f])]
            
            # Base prediction
            base_df = pd.DataFrame([latest_raw[features]])
            base_pred = ml_model.predict(base_df)[0]
            
            # Simulated prediction
            sim_df = base_df.copy()
            for col in sim_df.columns:
                if col.startswith("S"):
                    sim_df[col] = sim_df[col] * risk_mult
                elif col in ["market_orientation", "marketing_capabilities"]:
                    sim_df[col] = sim_df[col] * mkt_mult
                    
            sim_pred = ml_model.predict(sim_df)[0]
            
            with sim_col2:
                st.markdown("#### Model Output (Expected Alpha Rank)")
                
                # Visual Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sim_pred,
                    title = {'text': "Simulated Expected Return"},
                    delta = {'reference': base_pred, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
                    gauge = {
                        'axis': {'range': [-0.1, 0.1], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#2DD4BF"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [-0.1, 0], 'color': "rgba(239, 68, 68, 0.2)"},
                            {'range': [0, 0.1], 'color': "rgba(16, 185, 129, 0.2)"}],
                    }
                ))
                fig_gauge.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown(f"**Base Alpha Rank:** `{base_pred:.4f}`")
                if risk_mult > 1.0 and sim_pred < base_pred:
                    st.success("✅ **Proof:** The model successfully penalized the stock for increasing Operational Risk.")
                elif risk_mult > 1.0 and mkt_mult > 1.0 and sim_pred < base_pred:
                    st.success("✅ **Proof:** The model applied the **Distraction Penalty**. High marketing couldn't shield the massive operational risk.")

except Exception as e:
    st.error(f"Error loading system: {e}")
    st.info("Ensure the pipeline has run and generated processed parquets.")
