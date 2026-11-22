# Progress log

**Resume here:** read this file, then `docs/PLAN.md` (checklist) and `docs/DECISIONS.md`. Hard rules are in `../PROJECT_BRIEF.md` §1: never push or touch GitHub; the test window 2022-01 → 2025-06 stays locked until DECISIONS logs the unlock.

## Status by phase

| Phase | Status | Key outputs |
|---|---|---|
| 0 Environment, audit, plan | ✅ done 2026-11-22 | ENVIRONMENT.md, DATA_AUDIT.md, DECISIONS.md (audit + D-001…D-017), PLAN.md, pinned requirements |
| 1 Research grounding | 🔄 literature subagent running → docs/LITERATURE.md; RESEARCH_DESIGN.md pending | |
| 2 Point-in-time panel | 🔄 in progress | timing rules + tests ✅; risk loader ✅; marketing loader ✅; French factors/SIC maps ✅; SEC bulk + Yahoo downloads running |
| 3 Features | ⏳ | |
| 4 Models | ⏳ | |
| 5 Backtest | ⏳ | |
| 6 Robustness / SHAP | ⏳ | |
| 7 Website | ⏳ | |
| 8 Final QA | ⏳ | |

## Key numbers so far
- Risk scores: 131,920 filings, 16,959 CIKs, filed 2005-01 → 2025-01. `FYEAR` = filing year (100%). Filing dates match an original 10-K in EDGAR for 131,917/131,920.
- Marketing (v2): 337,599 GVKEY-quarters, 2003Q1–2025Q4. Linked to risk CIKs: **189,900 CIK-quarters, 4,804 firms** (5,289 ambiguous rows dropped). Raw constructs correlate 0.89–0.97 with each other.
- Quarter label = calendar quarter of the call (7/7 IPO checks).
- Yahoo universe: 4,535 risk CIKs with a current SEC ticker (3,746 on NYSE/Nasdaq) + SPY.
- Old current-ticker coverage of risk firm-years: 26% (2005) → 76% (2024).

## Next steps
1. When the downloads finish: parse submissions (SIC) and companyfacts (shares, book equity) → security master; build `prices_monthly`; check NVDA's June 2024 split against the market-cap math; sanity-replicate UMD/SMB.
2. Panel (Task 2.4) with no-look-ahead tests, then coverage funnel.
3. RESEARCH_DESIGN.md once LITERATURE.md lands.

## Session log
- 2026-11-22: Phase 0 complete. Old pipeline removed per the audit (see DECISIONS §A). New skeleton; Phase 2 data layer started.
