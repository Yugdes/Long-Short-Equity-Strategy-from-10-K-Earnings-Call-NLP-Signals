.PHONY: all data features backtest ml analysis app clean

all: data features backtest ml analysis

data:
	python -c "from src.data.ingest_risk import ingest_risk_scores; ingest_risk_scores()"
	python -c "from src.data.ingest_marketing import ingest_marketing_scores; ingest_marketing_scores()"
	python -c "from src.data.filings import build_filings_table; build_filings_table()"
	python -c "from src.data.security_master import build_security_master; build_security_master()"
	python -c "from src.data.factors import download_ff_factors; download_ff_factors()"

features:
	python -c "from src.features.signals import build_signals; build_signals()"
	python -c "from src.features.panel import build_stock_month_panel; build_stock_month_panel()"

backtest:
	python -c "from src.backtest.portfolios import compute_portfolio_returns; print('Run backtest scripts')"

ml:
	python -c "from src.ml.walkforward import walkforward_cv; print('Run ML pipeline')"

analysis:
	python -c "from src.analysis.diagnostics_text import run_text_diagnostics; print('Run Diagnostics')"

app:
	streamlit run app/streamlit_app.py

clean:
	python -c "import glob, os; [os.remove(f) for p in ['data/interim/*.parquet', 'data/processed/*.parquet', 'results/tables/*.csv', 'results/figures/*.png'] for f in glob.glob(p)]"
