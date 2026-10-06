"""
Main execution script.
Runs the entire pipeline end-to-end.
"""
import argparse
import sys
from pathlib import Path

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
    panel = build_stock_month_panel(signals, rets, factors, sec, marketing=mktg)
    
    # Phase 4: Backtest
    logger.info("--- Phase 4: Baseline Backtest ---")
    if not panel.empty and "S5_OpRisk_ind" in panel.columns:
        port_ret, port_stats = compute_portfolio_returns(
            panel, signal_col="S5_OpRisk_ind"
        )
        if not port_ret.empty:
            logger.info("Running factor attribution on L/S spread")
            regs = run_all_factor_models(port_ret["L/S"].dropna(), factors)
            logger.info(f"Alpha results:\n{regs[['model', 'alpha_annual', 't_alpha']]}")
            
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
            
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-run and ignore cache")
    args = parser.parse_args()
    
    run_pipeline(force=args.force)
