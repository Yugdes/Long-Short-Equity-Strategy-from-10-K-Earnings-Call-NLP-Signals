"""
Main execution script.
Runs the entire pipeline end-to-end.
"""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

from src.utils.helpers import setup_logging
from src.data.ingest_risk import ingest_risk_scores
from src.data.ingest_marketing import ingest_marketing_scores
from src.data.filings import build_filings_table
from src.data.security_master import build_security_master
from src.data.prices import build_returns_table
from src.data.factors import download_ff_factors
from src.features.signals import build_signals
from src.features.panel import build_stock_month_panel
from src.backtest.portfolios import compute_portfolio_returns
from src.backtest.factor_regs import run_all_factor_models
from src.analysis.diagnostics_text import run_text_diagnostics
from src.ml.walkforward import walkforward_cv
from src.ml.models import train_lightgbm
from src.ml.evaluation import evaluate_predictions

logger = setup_logging("run_all")

def run_pipeline(force: bool = False):
    logger.info("Starting pipeline execution")
    
    # Phase 1: Data
    logger.info("--- Phase 1: Data Ingestion ---")
    risk = ingest_risk_scores(force=force)
    mktg = ingest_marketing_scores(force=force)
    filings = build_filings_table(force=force)
    sec = build_security_master(risk_ciks=risk["cik"] if not risk.empty else None, force=force)
    
    # Note: Dummy tickers for the skeleton
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "WMT"] if not sec.empty else []
    rets = build_returns_table(tickers=tickers, force=force)
    factors = download_ff_factors(force=force)
    
    # Phase 2: Diagnostics
    logger.info("--- Phase 2: Diagnostics ---")
    out_dir = Path("results/figures")
    run_text_diagnostics(risk, sec, out_dir)
    
    # Phase 3: Features & Panel
    logger.info("--- Phase 3: Signals & Panel ---")
    signals = build_signals(risk, filings, sec)
    panel = build_stock_month_panel(signals, rets, factors, sec, marketing=mktg, force=force)
    
    # Phase 4: Backtest
    logger.info("--- Phase 4: Baseline Backtest ---")
    if not panel.empty and "S5_OpRisk_ind" in panel.columns:
        port_ret, port_stats = compute_portfolio_returns(
            panel, signal_col="S5_OpRisk_ind", min_names=1
        )
        if not port_ret.empty and "L/S" in port_ret.columns:
            logger.info("Running factor attribution on L/S spread")
            regs = run_all_factor_models(port_ret["L/S"].dropna(), factors)
            logger.info(f"Alpha results:\n{regs[['model', 'alpha_annual', 't_alpha']]}")
            
            # Save factor regression results to CSV
            out_dir = Path("results/tables")
            out_dir.mkdir(parents=True, exist_ok=True)
            regs.to_csv(out_dir / "factor_regressions.csv", index=False)
            logger.info(f"Saved factor regressions to {out_dir / 'factor_regressions.csv'}")
            
    # Phase 5: Machine Learning (Path B)
    logger.info("--- Phase 5: Machine Learning ---")
    if not panel.empty and "rank_excess_ret_next" in panel.columns:
        features = [c for c in panel.columns if c.startswith("S")] + \
                   ["market_orientation", "marketing_capabilities", "marketing_excellence"]
        # Only keep features that actually exist in the panel and aren't completely null
        features = [f for f in features if f in panel.columns and panel[f].notna().sum() > 0]
        
        logger.info(f"Running Walk-Forward LightGBM with {len(features)} features")
        preds, models = walkforward_cv(
            panel, features, target="rank_excess_ret_next", model_func=train_lightgbm
        )
        
        if not preds.empty:
            eval_metrics = evaluate_predictions(preds)
            logger.info(f"ML Evaluation Metrics: {eval_metrics}")
            
            # Save predictions
            preds.to_parquet("data/processed/ml_predictions.parquet")
            
            # Save models
            model_dir = Path("results/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(models):
                joblib.dump(model, model_dir / f"lightgbm_split_{i}.joblib")
            logger.info(f"Saved {len(models)} ML models to {model_dir}")
            
            # Plot Model Results / Feature Importance (if model has feature_importances_)
            import matplotlib.pyplot as plt
            import seaborn as sns
            if hasattr(models[-1], "feature_importances_"):
                imp = models[-1].feature_importances_
                df_imp = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df_imp.head(15), x="importance", y="feature")
                plt.title("Top 15 Feature Importances (Latest Model Split)")
                plt.tight_layout()
                plt.savefig("results/figures/feature_importances.png")
                plt.close()

            # Plot Risk vs Marketing relationship
            logger.info("Generating Risk vs Marketing plots")
            plot_df = panel.dropna(subset=["S1_OpRisk", "market_orientation"]).copy()
            if not plot_df.empty:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=plot_df.sample(min(10000, len(plot_df))), x="market_orientation", y="S1_OpRisk", alpha=0.1)
                sns.regplot(data=plot_df.sample(min(10000, len(plot_df))), x="market_orientation", y="S1_OpRisk", scatter=False, color="red")
                plt.title("Operational Risk vs Marketing Orientation")
                plt.xlabel("Marketing Orientation Score")
                plt.ylabel("Operational Risk Score")
                plt.tight_layout()
                plt.savefig("results/figures/risk_vs_marketing.png")
                plt.close()
                logger.info("Saved Risk vs Marketing plot")
            
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-run and ignore cache")
    args = parser.parse_args()
    
    run_pipeline(force=args.force)
