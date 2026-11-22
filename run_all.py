"""One command: raw data -> panel -> features -> tests -> models -> backtests -> robustness -> website data.

    python run_all.py                 # run every stage (each stage reuses its cache if present)
    python run_all.py --from features # rerun from a stage onward (its caches are rebuilt)
    python run_all.py --only export   # one stage

Downloads (Yahoo, SEC bulk, Ken French) are cached under data/raw/ and never re-fetched once present,
which freezes the free-data vintage. Delete data/raw/{yahoo,sec,french} to refresh them.
"""
import argparse
import subprocess
import sys
import time

STAGES = ["download", "panel", "features", "tests", "single", "ml", "analysis", "export", "notebooks"]


def stage_download(force):
    from scripts.download_prices import main as dl
    from src.data.sec import download
    for k in ("tickers", "submissions", "companyfacts"):
        download(k)
    dl()


def stage_panel(force):
    if force:  # rebuild every cache derived from prices (D-028): stale caches must never survive a rerun
        import shutil
        from src.utils.config import path
        for f in ["prices_monthly.parquet", "splits.parquet"]:
            (path("interim") / f).unlink(missing_ok=True)
        for f in path("processed").glob("*.parquet"):
            f.unlink()
        shutil.rmtree(path("processed") / "preds", ignore_errors=True)
        (path("results") / "spec_log.csv").unlink(missing_ok=True)
    from src.features.panel import build_panel
    from src.evaluation.sanity import factor_replication, coverage_funnel
    p = build_panel(force=force)
    print(factor_replication(p).round(3).to_string())
    coverage_funnel(p)


def stage_features(force):
    import pandas as pd
    from src.features.signals import make_features
    from src.utils.config import path
    p = pd.read_parquet(path("processed") / "panel.parquet")
    for mask, name in (("elig", "features.parquet"), ("elig_ext", "features_ext.parquet")):
        f = path("processed") / name
        if force or not f.exists():
            make_features(p, mask).to_parquet(f, index=False)


def stage_tests(force):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"])
    if r.returncode:
        raise SystemExit("tests failed: fix before producing results")


def stage_single(force):
    import pandas as pd
    from src.evaluation.single_signals import (run_single_signals, run_fama_macbeth, run_vol_prediction,
                                               run_interaction_fm)
    from src.utils.config import path
    f = pd.read_parquet(path("processed") / "features.parquet")
    run_single_signals(f), run_fama_macbeth(f), run_vol_prediction(f), run_interaction_fm(f)


def stage_ml(force):
    from src.models.run_ml import main
    from src.models.diagnostics import main as diagnostics
    main(horizon=1), main(horizon=3), diagnostics()
    from src.utils.lock import is_unlocked
    if is_unlocked():
        main(horizon=1, final=True)


def stage_analysis(force):
    from src.viz.export import run_analysis
    run_analysis()


def stage_export(force):
    from src.viz.export import export_json
    export_json()


def stage_notebooks(force):
    subprocess.run([sys.executable, "scripts/make_notebooks.py"], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STAGES)
    ap.add_argument("--only", choices=STAGES)
    a = ap.parse_args()
    todo = [a.only] if a.only else STAGES[STAGES.index(a.start):] if a.start else STAGES
    for s in todo:
        t = time.time()
        print(f"=== {s} ===", flush=True)
        globals()[f"stage_{s}"](force=bool(a.start or a.only))
        print(f"=== {s} done in {time.time() - t:.0f}s ===", flush=True)
