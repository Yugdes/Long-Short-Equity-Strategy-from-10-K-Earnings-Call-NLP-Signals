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
    panel = build_stock_month_panel(signals, rets, factors, sec)
    
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
            
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-run and ignore cache")
    args = parser.parse_args()
    
    run_pipeline(force=args.force)
