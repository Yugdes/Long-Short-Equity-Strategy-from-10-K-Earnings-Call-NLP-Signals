# Environment & feasibility (Phase 0)

Checked 2026-11-22 on the project laptop.

## Hardware / OS

| Item | Value |
|---|---|
| CPU | Intel Core i5-12450H (12th gen), 8 physical cores (4P + 4E), 12 logical threads, 2.0 GHz base |
| RAM | 15.7 GB total; only **~2.3 GB free** at check time (VS Code, browser and other apps resident) |
| Disk (D:, project drive) | 189.9 GB free (54.9 GB used) |
| GPU | NVIDIA GeForce GTX 1650, 4 GB VRAM, driver 592.82 (+ Intel UHD iGPU) |
| OS | Windows 11 Home 10.0.26300 |
| Python | 3.10.11 (project venv `venv/`) |
| Node / npm | v24.21.0 / 11.19.0 (for the Vite website) |

## Network reachability (all HTTP 200)

SEC `www.sec.gov` (ticker/exchange file, bulk archives), `data.sec.gov` (XBRL, submissions), Kenneth French data library (FF5, momentum, `Siccodes48.zip`), Yahoo Finance via `yfinance` (daily OHLCV plus dividends and splits).

## Python environment

- Isolated venv at `venv/` (Python 3.10.11), reused from earlier work rather than recreated. It already holds every package the plan needs, and rebuilding it gains nothing. Logged in DECISIONS.md.
- `requirements.txt`: pinned **direct** dependencies. `requirements-lock.txt`: full `pip freeze` (162 packages) for exact reproduction.
- Recreate from scratch:
  ```
  py -3.10 -m venv venv
  venv\Scripts\python -m pip install -r requirements-lock.txt
  ```

## What is feasible on this machine

| Workload | Size | Verdict |
|---|---|---|
| Risk scores (132k firm-years), marketing scores (338k firm-quarters) | small | Trivial. Read the Excel file once and cache it as parquet. |
| Stock-month panel | ~0.6–0.9M rows × ~80 features | OK as float32 (~0.3 GB). Build per year or column group to respect the ~2–8 GB of free RAM. |
| yfinance daily download (~7–8k tickers, 2004→) | network-bound, ~20–30 min | OK. Batches of 50, cached per batch, resumable. |
| SEC bulk `companyfacts.zip` / `submissions.zip` | ~1–2 GB each, zipped | OK. Stream the zip and parse only the needed CIKs and concepts; never fully extract. |
| Portfolio sorts, Fama-MacBeth, Newey-West, bootstraps, placebo permutations | CPU, seconds to minutes | OK |
| ElasticNet / LightGBM / XGBoost walk-forward (≈12 annual refits, small tuning grid) | ~1M rows, 8 threads | OK (≈ tens of minutes per model family). Use `n_jobs=6`, `max_bin=63`, early stopping. |
| Small MLP (2 hidden layers) | ~1M rows × ~80 features | OK on CPU with scikit-learn `MLPRegressor` (early stopping). PyTorch/CUDA is not needed and is not installed. |
| SHAP (TreeExplainer) | ≤ 50k-row sample | OK, minutes |
| Transformer fine-tuning or re-scoring raw 10-K / transcript text | ~10⁸ sentences | **Not planned.** Not needed, because the papers supply the scores. Transcripts are licensed (FactSet) and not available. |

Practical notes: run long jobs on mains power, and write a checkpoint after each walk-forward refit. Every multiprocessing entry point is guarded with `if __name__ == "__main__":` (Windows spawn).
