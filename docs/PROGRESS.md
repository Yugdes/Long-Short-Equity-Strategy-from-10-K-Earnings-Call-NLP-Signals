# Progress log

**Resume here:** read this file, then `docs/PLAN.md` (checklist) and `docs/DECISIONS.md`. Hard rules are in `../PROJECT_BRIEF.md` §1: never push or touch GitHub; the test window 2022-01 → 2025-06 stays locked until DECISIONS logs the unlock.

## Status by phase

| Phase | Status | Key outputs |
|---|---|---|
| 0 Environment, audit, plan | ✅ done 2026-11-22 | ENVIRONMENT.md, DATA_AUDIT.md, DECISIONS.md (audit + D-001…D-017), PLAN.md, pinned requirements |
| 1 Research grounding | ✅ done 2026-11-22 | LITERATURE.md (16 verified works), RESEARCH_DESIGN.md (H1–H5, primary spec, hurdles) |
| 2 Point-in-time panel | ✅ done 2026-11-22 | data/processed/panel.parquet (308,283 rows; 222k primary stock-months); 32 tests pass incl. no-look-ahead invariants on the real panel |
| 3 Features | ✅ code done (src/features/signals.py, tests) | ~90 rank features in 4 groups |
| 4 Models | ✅ dev + locked test (rerun after D-028/D-029) | results/tables/ml_summary_{dev,final}_h1.csv |
| 5 Backtest | ✅ | results/json/strategy.json (dev + test tear sheets) |
| 6 Robustness / SHAP | ✅ | subperiods, size, industries, alt lags, h=3, placebo, shuffled labels, DSR, PBO, SHAP, low-vol factor |
| 7 Website | ✅ | website/ (npm run dev / build); Playwright: desktop 1440 + mobile 390 + dark, no console errors |
| 8 Final QA | ✅ | clean `run_all.py --from panel` end to end; 38 tests pass; leakage review (D-029) and site/docs-vs-code review (D-030) applied; README headline generated from JSON |

## Key numbers so far
- Risk scores: 131,920 filings, 16,959 CIKs, filed 2005-01 → 2025-01. `FYEAR` = filing year (100%). Filing dates match an original 10-K in EDGAR for 131,917/131,920.
- Marketing (v2): 337,599 GVKEY-quarters, 2003Q1–2025Q4. Linked to risk CIKs: **189,900 CIK-quarters, 4,804 firms** (5,289 ambiguous rows dropped). Raw constructs correlate 0.89–0.97 with each other.
- Quarter label = calendar quarter of the call (7/7 IPO checks).
- Yahoo universe: 4,535 risk CIKs with a current SEC ticker (3,746 on NYSE/Nasdaq) + SPY.
- Old current-ticker coverage of risk firm-years: 26% (2005) → 76% (2024).

## Phase 2 numbers
- Yahoo: 4,456/4,536 tickers with data; 750,424 ticker-months; 548 returns blanked as errors (D-020).
- Primary universe: 2010-01 → 2025-06, median 1,133 stocks/month (min 195 in early 2010, max 1,725). Extended EW universe from 2006-07.
- Coverage funnel (`results/tables/coverage_funnel.csv`): firms in the primary universe per filing year = 929 (2012) → 1,899 (2024), i.e. 12–33% of all risk filings.
- Sanity: our momentum decile L/S vs French UMD corr 0.79; size L/S vs SMB corr 0.81 (2010–2021).
- Market-cap fixes D-021, capped VW D-022.

## Phase 4 numbers (dev window, 2010–2021 for sorts / FM)
- 42 single-signal decile sorts (EW+VW): **none clears |t| ≥ 3**. Best: Δ marketing excellence YoY VW α t = 2.91; non-op risk level EW L/S t = 2.69 (high risk earns *more*: risk-premium reading).
- H2 (risk increases → lower returns): null (|t| < 1.8 everywhere).
- H4 interaction FM: null (max t = 0.89). 3×3 double sort (D-031): VW cushion α t −2.2 to −1.0 for risk level (wrong sign), +1.7 to +2.3 for risk change (right sign, below hurdle) → H4 "Below hurdle".
- H1(a) volatility: risk text predicts next-12m volatility after controls: legal t = 5.7, marketing 3.3, technology 2.7 (higher vol); accounting −11.8, management −5.8, finance −3.7 (lower vol).

## ML ladder (dev OOS 2014–2021, 96 months)
- Every model has rank IC 0.030–0.048 with NW t > 3 (controls-only LGBM 0.039; ALL LGBM 0.048; ens ALL 0.0446 vs ens C 0.0409).
- Risk text adds ~0 IC over controls; marketing-call language adds ~+0.01 (D-026: not the coverage flag).
- **No decile L/S has a significant FF5+Mom alpha** (VW |t| ≤ 1.1; EW ≤ 1.6). Primary (ens ALL VW d10) gross Sharpe 0.08, α −3.3%/yr (t −0.59).
- FM of returns on the forecast rank with controls: t ≈ 2.5–2.7 (below the hurdle).

## Test unlock, bug and rerun (2026-11-22)
- D-027: design frozen, test window unlocked. The first test run looked too good (VW Sharpe 1.5–2.0, α t up to 5.3).
- D-028: cause was look-ahead from reverse splits in 2026 missing from the Yahoo split list (end-dated download). Fixed: download with no end date; everything rebuilt.
- D-029: independent reviewer findings fixed (rank ensemble, sample end 2025-05, h3 embargo, cleaning rule, DSR family, continuous tear sheet, raw-ZCA robustness, time-varying survivorship caveat).
- Corrected dev (2014–2021): ens ALL IC 0.041 (t 3.6), VW d10 gross Sharpe 0.28, FF5+Mom α t 0.37. Corrected test (2022-01–2025-05): IC 0.048–0.067, VW gross Sharpe −0.03–1.2, α t −2.5 to 1.0; text adds nothing over controls in test.

## Final numbers (definitive run)
See README headline block (generated from results/json). Dev IC 0.041 (t 3.6), test IC 0.060 (t 4.5; controls-only 0.064). Primary net Sharpe −0.12 dev / 0.51 test; net FF5+Mom α t −0.85 / −0.83. DSR 0.08 (123 specs, after D-031's 24 double-sort trials; 0.11 over 99 before), PBO 0.88. Verdict: prediction yes, alpha no; text adds little beyond known characteristics; risk text predicts volatility (5/8 factors, both signs).

## Supplementary studies (2026-11-23, D-032..D-034; `run_all.py --only studies`)
- Survivorship audit: coverage of once-listed filers 26% (2006) → 80% (2024); low-risk decile loses the most firms (70% vs 55%); missing firms would need +1.7%/month to erase the VW aggregate-risk spread. Finance risk predicts bankruptcy within 3y (t = 8.6), operations t = 4.6.
- Filing-date event study (15,797 10-Ks): legal risk → larger |CAR[0,+1]| (t = 5.1); no post-filing drift (|t| < 1.3).
- Replication of the source paper: operations → next-year ln(COGS) 0.289 (SE 0.107) vs 0.247 (0.041); interaction not replicated.

## Next steps (open items for the user)
1. Final report written: `report/main.pdf` (28 pages; source `report/main.tex`, assets generated by `report/make_report_assets.py`, stage `report` in run_all.py).
2. Website names filled in (authors; supervisor Professor Marcos Severo).
3. Push to GitHub yourself (nothing has been pushed; the `origin` remote was left untouched).
4. Optional upgrade: a CRSP/Compustat extract would remove the survivorship bias (see Limitations).

## Session log
- 2026-11-22: Phase 0 complete. Old pipeline removed per the audit (see DECISIONS §A). New skeleton; Phase 2 data layer started.
- 2026-11-22: Phases 1–8 complete. Test unlocked (D-027); split look-ahead bug found and fixed (D-028); two independent reviews applied (D-029, D-030); definitive clean rerun.
- 2026-11-23: Added the pre-registered H4 3×3 double sort (D-031): code, test, site table and ledger; reran single → analysis → export → notebooks.
- 2026-11-23: Supervisor named on site and README. Added survivorship audit, filing-date event study and source-paper replication (D-032..D-034): code, tests, `studies` stage, site section "Further evidence", README rows.
- 2026-11-23: Academic LaTeX report written (`report/`): generator for 172 number macros, 17 tables, 8 figures; compiled via the new `report` stage.
