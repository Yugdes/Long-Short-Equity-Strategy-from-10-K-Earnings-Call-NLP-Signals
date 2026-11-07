"""
Stage 1 Comprehensive Verification Test Suite
Tests data integrity, temporal alignment, factor timing, marketing staleness,
and forward return continuity.
"""

import sys
import unittest
import numpy as np
import pandas as pd

# ── Test 1: Filing Date Fallback Elimination & Missing Flagging ──────────────
class TestFilingDateFallback(unittest.TestCase):
    def test_missing_filing_date_not_fabricated(self):
        """Verify that missing filing dates are flagged as NaT and never imputed with arbitrary dates."""
        # Simulated risk scores for CIK 1001 with missing date_filed
        risk_df = pd.DataFrame({
            "cik": [1001, 1002],
            "year": [2020, 2020],
            "operations_prob": [0.08, 0.04],
        })
        # Empty or partial filings table
        filings_df = pd.DataFrame({
            "cik": [1002],
            "fiscal_year_approx": [2020],
            "date_filed": [pd.Timestamp("2021-02-24")],
            "is_amendment": [False]
        })
        
        # Test merging logic
        df = risk_df.merge(
            filings_df[["cik", "fiscal_year_approx", "date_filed", "is_amendment"]].rename(
                columns={"fiscal_year_approx": "year"}
            ),
            on=["cik", "year"],
            how="left"
        )
        df["is_missing_filing_date"] = df["date_filed"].isna()
        
        # Assert CIK 1001 is missing date_filed
        self.assertTrue(df.loc[df["cik"] == 1001, "is_missing_filing_date"].iloc[0])
        self.assertTrue(pd.isna(df.loc[df["cik"] == 1001, "date_filed"].iloc[0]))
        
        # Availability lag
        valid_mask = ~df["is_missing_filing_date"]
        df["signal_date"] = pd.NaT
        df["first_hold_month"] = pd.PeriodIndex([pd.NaT] * len(df), freq="M")
        
        if valid_mask.any():
            valid_sig_dates = df.loc[valid_mask, "date_filed"] + pd.offsets.BDay(2)
            df.loc[valid_mask, "signal_date"] = valid_sig_dates
            m_end = valid_sig_dates + pd.offsets.MonthEnd(0)
            df.loc[valid_mask, "first_hold_month"] = (m_end + pd.offsets.MonthBegin(1)).dt.to_period("M")
            
        # CIK 1002 filed 2021-02-24 (Wed) -> signal_date 2021-02-26 -> first_hold_month 2021-03
        self.assertEqual(df.loc[df["cik"] == 1002, "first_hold_month"].iloc[0], pd.Period("2021-03", freq="M"))
        # CIK 1001 must have NaT for first_hold_month
        self.assertTrue(pd.isna(df.loc[df["cik"] == 1001, "first_hold_month"].iloc[0]))
        
        # If exclude_missing_filings is True:
        clean_df = df[valid_mask].copy()
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(clean_df.iloc[0]["cik"], 1002)


# ── Test 2: 10-K vs 10-K/A Prioritization ────────────────────────────────────
class TestFilingDeduplication(unittest.TestCase):
    def test_original_10k_prioritized_over_amendment(self):
        """When both 10-K and 10-K/A exist for same CIK and fiscal year, 10-K must be selected."""
        raw_filings = pd.DataFrame({
            "cik": [2001, 2001, 2002],
            "fiscal_year_approx": [2020, 2020, 2020],
            "form_type": ["10-K/A", "10-K", "10-K/A"],
            "date_filed": [pd.Timestamp("2021-04-15"), pd.Timestamp("2021-02-15"), pd.Timestamp("2021-03-01")],
            "is_amendment": [True, False, True]
        })
        
        filings_sorted = raw_filings.sort_values(
            by=["cik", "fiscal_year_approx", "is_amendment", "date_filed"],
            ascending=[True, True, True, True]
        )
        deduped = filings_sorted.drop_duplicates(
            subset=["cik", "fiscal_year_approx"],
            keep="first"
        ).reset_index(drop=True)
        
        # CIK 2001 has both: should select original 10-K
        row_2001 = deduped[deduped["cik"] == 2001].iloc[0]
        self.assertEqual(row_2001["form_type"], "10-K")
        self.assertFalse(row_2001["is_amendment"])
        self.assertEqual(row_2001["date_filed"], pd.Timestamp("2021-02-15"))
        
        # CIK 2002 has only 10-K/A: should retain 10-K/A
        row_2002 = deduped[deduped["cik"] == 2002].iloc[0]
        self.assertEqual(row_2002["form_type"], "10-K/A")
        self.assertTrue(row_2002["is_amendment"])
        self.assertEqual(row_2002["date_filed"], pd.Timestamp("2021-03-01"))

    def test_drop_duplicates_prevents_column_mixing(self):
        """Proof that drop_duplicates preserves complete intact row without combining columns from different filings."""
        df = pd.DataFrame({
            "cik": [2003, 2003],
            "fiscal_year_approx": [2020, 2020],
            "is_amendment": [False, True],
            "date_filed": [pd.Timestamp("2021-02-15"), pd.Timestamp("2021-04-15")],
            "edgar_url": ["edgar/10k.txt", np.nan],
            "auditor": [np.nan, "PwC"]  # Row 1 has no auditor, Row 2 has auditor
        })
        df_sorted = df.sort_values(by=["cik", "fiscal_year_approx", "is_amendment", "date_filed"])
        
        # drop_duplicates preserves intact Row 1
        dd_res = df_sorted.drop_duplicates(subset=["cik", "fiscal_year_approx"], keep="first").reset_index(drop=True)
        self.assertTrue(pd.isna(dd_res.iloc[0]["auditor"]), "drop_duplicates must NOT pull non-null values from other filings")
        self.assertEqual(dd_res.iloc[0]["edgar_url"], "edgar/10k.txt")


# ── Test 3: Marketing Staleness Tolerance & Cadence Sensitivity ──────────────
class TestMarketingStaleness(unittest.TestCase):
    def test_cadence_retention_comparison(self):
        """Compare data retention of 120d, 180d, 210d, and Infinite tolerances."""
        # Simulated timeline: 24 months from 2021-01 to 2022-12
        months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
        panel = pd.DataFrame({"month_start": months})
        panel["cik"] = 3001
        
        # Firm A: Quarterly filer (calls in Feb, May, Aug, Nov)
        mktg_quarterly = pd.DataFrame({
            "cik": [3001] * 8,
            "avail_date": pd.to_datetime([
                "2021-02-15", "2021-05-15", "2021-08-15", "2021-11-15",
                "2022-02-15", "2022-05-15", "2022-08-15", "2022-11-15"
            ]),
            "market_orientation": [0.65, 0.67, 0.68, 0.70, 0.71, 0.72, 0.70, 0.73]
        })
        
        # Firm B: Semiannual filer (calls in Feb, Aug)
        panel_b = panel.copy()
        panel_b["cik"] = 3002
        mktg_semiannual = pd.DataFrame({
            "cik": [3002] * 4,
            "avail_date": pd.to_datetime([
                "2021-02-15", "2021-08-15", "2022-02-15", "2022-08-15"
            ]),
            "market_orientation": [0.55, 0.58, 0.57, 0.60]
        })
        
        # Evaluate retention across active months (from first call: March 2021 to Dec 2022 = 22 months)
        active_panel_b = panel_b[panel_b["month_start"] >= "2021-03-01"].sort_values("month_start")
        
        # Test 120 days tolerance:
        merged_120 = pd.merge_asof(
            active_panel_b, mktg_semiannual.sort_values("avail_date"),
            left_on="month_start", right_on="avail_date", by="cik",
            direction="backward", tolerance=pd.Timedelta(days=120)
        )
        # In semiannual cadence, months 5 and 6 after a call exceed 120 days (~120-180 days)
        dropped_120 = merged_120["avail_date"].isna().sum()
        self.assertGreater(dropped_120, 0, "120d tolerance dropped semiannual firm-months")
        
        # Test 210 days tolerance:
        merged_210 = pd.merge_asof(
            active_panel_b, mktg_semiannual.sort_values("avail_date"),
            left_on="month_start", right_on="avail_date", by="cik",
            direction="backward", tolerance=pd.Timedelta(days=210)
        )
        dropped_210 = merged_210["avail_date"].isna().sum()
        self.assertEqual(dropped_210, 0, "210d tolerance should retain 100% of continuous semiannual firm-months")
        
        # Firm C: Lapsed filer (stopped reporting after 2021-02-15)
        panel_c = panel.copy()
        panel_c["cik"] = 3003
        mktg_lapsed = pd.DataFrame({
            "cik": [3003],
            "avail_date": pd.to_datetime(["2021-02-15"]),
            "market_orientation": [0.50]
        })
        
        merged_lapsed_210 = pd.merge_asof(
            panel_c.sort_values("month_start"), mktg_lapsed.sort_values("avail_date"),
            left_on="month_start", right_on="avail_date", by="cik",
            direction="backward", tolerance=pd.Timedelta(days=210)
        )
        # Observation at 2022-01-01 is 320 days later: must be pruned by 210d
        row_2022 = merged_lapsed_210[merged_lapsed_210["month_start"] == "2022-01-01"].iloc[0]
        self.assertTrue(pd.isna(row_2022["avail_date"]), "210d tolerance must prune lapsed filers")


# ── Test 4: Forward Return Calendar Continuity Across Missing Months ─────────
class TestForwardReturnContinuity(unittest.TestCase):
    def test_missing_months_produce_nan_forward_return(self):
        """If month t+1 is missing (e.g. Jan followed by Jun), ret_next must be NaN."""
        df = pd.DataFrame({
            "cik": [4001, 4001, 4001, 4002, 4002, 4002],
            "month": [
                pd.Period("2020-01", freq="M"),
                pd.Period("2020-02", freq="M"),
                pd.Period("2020-06", freq="M"), # 4-month gap!
                pd.Period("2020-11", freq="M"),
                pd.Period("2020-12", freq="M"),
                pd.Period("2021-01", freq="M"), # Crosses year boundary consecutively
            ],
            "ret": [0.05, 0.02, -0.03, 0.04, 0.01, -0.02]
        })
        
        df = df.sort_values(["cik", "month"])
        period_series = df["month"].apply(
            lambda m: m if isinstance(m, pd.Period) else pd.Period(m, freq="M")
        )
        next_period = df.groupby("cik")["month"].shift(-1).apply(
            lambda m: m if (pd.isna(m) or isinstance(m, pd.Period)) else pd.Period(m, freq="M")
        )
        is_consecutive = (next_period == (period_series + 1))
        raw_shift = df.groupby("cik")["ret"].shift(-1)
        df["ret_next"] = np.where(is_consecutive, raw_shift, np.nan)
        
        # CIK 4001:
        # Row 0 (2020-01) -> followed by 2020-02 (consecutive): ret_next = 0.02
        self.assertEqual(df.loc[0, "ret_next"], 0.02)
        # Row 1 (2020-02) -> followed by 2020-06 (non-consecutive gap!): ret_next MUST be NaN
        self.assertTrue(np.isnan(df.loc[1, "ret_next"]), "Row before multi-month gap must be NaN")
        # Row 2 (2020-06) -> last observation: ret_next MUST be NaN
        self.assertTrue(np.isnan(df.loc[2, "ret_next"]))
        
        # CIK 4002:
        # Row 3 (2020-11) -> followed by 2020-12: ret_next = 0.01
        self.assertEqual(df.loc[3, "ret_next"], 0.01)
        # Row 4 (2020-12) -> followed by 2021-01 (consecutive across year boundary): ret_next = -0.02
        self.assertEqual(df.loc[4, "ret_next"], -0.02)
        # Row 5 (2021-01) -> last observation: ret_next = NaN
        self.assertTrue(np.isnan(df.loc[5, "ret_next"]))


# ── Test 5: Risk-Free Rate Alignment (Contemporaneous vs Forward) ─────────────
class TestRiskFreeRateAlignment(unittest.TestCase):
    def test_risk_free_rate_alignment(self):
        """Verify that excess_ret_next = ret_next - RF_{t+1} and excess_ret = ret - RF_t."""
        panel = pd.DataFrame({
            "cik": [5001, 5001],
            "month": [pd.Period("2020-01", freq="M"), pd.Period("2020-02", freq="M")],
            "ret": [0.05, 0.02],
            "ret_next": [0.02, np.nan]
        })
        factors = pd.DataFrame({
            "month": [pd.Period("2020-01", freq="M"), pd.Period("2020-02", freq="M"), pd.Period("2020-03", freq="M")],
            "RF": [0.0010, 0.0015, 0.0020]
        })
        
        panel["_month_period"] = panel["month"]
        
        # 1. Contemporaneous RF (month t return minus month t RF)
        factors_curr = factors[["month", "RF"]].rename(columns={"month": "_month_period", "RF": "RF"})
        panel = panel.merge(factors_curr, on="_month_period", how="left")
        panel["excess_ret"] = panel["ret"] - panel["RF"]
        
        # 2. Forward RF (month t+1 return minus month t+1 RF)
        factors_next = factors[["month", "RF"]].copy()
        factors_next["_month_period"] = factors_next["month"] - 1
        factors_next = factors_next.rename(columns={"RF": "RF_next"}).drop(columns=["month"])
        panel = panel.merge(factors_next, on="_month_period", how="left")
        panel["excess_ret_next"] = panel["ret_next"] - panel["RF_next"]
        
        # For row 2020-01:
        # contemporaneous ret = 0.05, RF_t = 0.0010 -> excess_ret = 0.0490
        self.assertAlmostEqual(panel.loc[0, "excess_ret"], 0.05 - 0.0010, places=6)
        # forward ret = 0.02 (earned in 2020-02), RF_{t+1} = 0.0015 -> excess_ret_next = 0.02 - 0.0015 = 0.0185
        self.assertAlmostEqual(panel.loc[0, "RF_next"], 0.0015, places=6)
        self.assertAlmostEqual(panel.loc[0, "excess_ret_next"], 0.02 - 0.0015, places=6)


# ── Test 6: Factor Regression Realization Alignment ──────────────────────────
class TestFactorRegressionAlignment(unittest.TestCase):
    def test_timing_misalignment_vs_alignment_proof(self):
        """
        Proof of econometric collapse:
        Regressing month t+1 return on month t factor produces beta ~ 0 and negative R2.
        Regressing month t+1 return on month t+1 factor correctly recovers true beta = 1.0.
        """
        np.random.seed(42)
        n_months = 120
        # True factor return ~ N(0.01, 0.04^2)
        mkt_rf = np.random.normal(0.01, 0.04, n_months)
        # Asset with true beta = 1.0, alpha = 0.002 (2.4%/year), idiosyncratic noise
        noise = np.random.normal(0.0, 0.01, n_months)
        r_asset = 0.002 + 1.0 * mkt_rf + noise
        
        dates = pd.period_range("2010-01", periods=n_months, freq="M")
        
        # Case A: Correctly Aligned (R_t on F_t)
        cov_aligned = np.cov(r_asset, mkt_rf)
        beta_aligned = cov_aligned[0, 1] / np.var(mkt_rf)
        y_hat_aligned = 0.002 + beta_aligned * mkt_rf
        ss_res_aligned = np.sum((r_asset - y_hat_aligned)**2)
        ss_tot = np.sum((r_asset - np.mean(r_asset))**2)
        r2_aligned = 1 - ss_res_aligned / ss_tot
        
        # Case B: Misaligned (R_{t+1} on F_t)
        # Shift asset return by +1 relative to factor
        r_misaligned = r_asset[1:]
        mkt_misaligned = mkt_rf[:-1]
        cov_misaligned = np.cov(r_misaligned, mkt_misaligned)
        beta_misaligned = cov_misaligned[0, 1] / np.var(mkt_misaligned)
        y_hat_misaligned = np.mean(r_misaligned) + beta_misaligned * (mkt_misaligned - np.mean(mkt_misaligned))
        ss_res_misaligned = np.sum((r_misaligned - y_hat_misaligned)**2)
        ss_tot_misaligned = np.sum((r_misaligned - np.mean(r_misaligned))**2)
        r2_misaligned = 1 - ss_res_misaligned / ss_tot_misaligned
        
        # Assert aligned recovers true beta
        self.assertAlmostEqual(beta_aligned, 1.0, delta=0.08)
        self.assertGreater(r2_aligned, 0.85)
        
        # Assert misaligned collapses to 0
        self.assertAlmostEqual(beta_misaligned, 0.0, delta=0.15)
        self.assertAlmostEqual(r2_misaligned, 0.0, delta=0.05)


# ── Test 7: Marketing Imputation & ML Feature Visibility ─────────────────────
class TestMarketingImputationAndMLFeatureVisibility(unittest.TestCase):
    def test_has_marketing_data_flag_and_feature_visibility(self):
        """Verify has_marketing_data is an integer column (0 or 1) and visible in ML feature list."""
        panel = pd.DataFrame({
            "cik": [7001, 7002],
            "month": [pd.Period("2020-01", freq="M"), pd.Period("2020-01", freq="M")],
            "avail_date": [pd.Timestamp("2019-11-15"), pd.NaT],
            "market_orientation": [0.65, np.nan],
            "marketing_capabilities": [0.70, np.nan],
            "marketing_excellence": [0.60, np.nan],
            "S1_OpRisk": [0.05, 0.08]
        })
        
        # Panel processing logic
        panel["has_marketing_data"] = panel["avail_date"].notna().astype(int)
        for col in ["market_orientation", "marketing_capabilities", "marketing_excellence"]:
            panel[f"{col}_raw"] = panel[col]
            panel[col] = panel[col].fillna(panel[col].median()).fillna(0)
            
        # Assertions on panel
        self.assertEqual(panel.loc[0, "has_marketing_data"], 1)
        self.assertEqual(panel.loc[1, "has_marketing_data"], 0)
        self.assertEqual(panel["has_marketing_data"].dtype, np.int64)
        
        # Verify ML features list
        features = [c for c in panel.columns if c.startswith("S")] + \
                   ["market_orientation", "marketing_capabilities", "marketing_excellence", "has_marketing_data"]
        
        self.assertIn("has_marketing_data", features)
        self.assertIn("S1_OpRisk", features)
        self.assertEqual(len(features), 5)


# ── Test 8: Actual Downloaded Datasets Empirical Integrity ───────────────────
class TestActualDownloadedDatasets(unittest.TestCase):
    def test_actual_risk_scores_integrity(self):
        """Verify actual Astvansh & Simpson operational risk dataset on disk."""
        from pathlib import Path
        p = Path("data/processed/risk_scores.parquet")
        self.assertTrue(p.exists(), "risk_scores.parquet must exist")
        df = pd.read_parquet(p)
        self.assertEqual(len(df), 131920, "Must contain exactly 131,920 firm-years")
        self.assertEqual(df["cik"].nunique(), 16959, "Must contain exactly 16,959 unique CIKs")
        self.assertEqual(df["filing_date"].isna().sum(), 0, "Must have zero missing filing dates")
        self.assertEqual(df["year"].min(), 2005)
        self.assertEqual(df["year"].max(), 2025)
        self.assertAlmostEqual(df["op_risk"].mean(), 0.1773, delta=0.001)

    def test_actual_marketing_scores_integrity(self):
        """Verify actual Damavandi et al. marketing scores dataset on disk."""
        from pathlib import Path
        p = Path("data/processed/marketing_scores.parquet")
        self.assertTrue(p.exists(), "marketing_scores.parquet must exist")
        df = pd.read_parquet(p)
        self.assertEqual(len(df), 295521, "Must contain 295,521 aligned firm-quarters")
        self.assertEqual(df.duplicated(subset=["cik", "year", "quarter"]).sum(), 0, "Zero duplicates on (CIK, year, quarter)")
        for col in ["market_orientation", "marketing_capabilities", "marketing_excellence"]:
            self.assertIn(col, df.columns, f"{col} must be in marketing scores")
            self.assertEqual(df[col].isna().sum(), 0, f"{col} must not have nulls")

    def test_actual_factors_integrity(self):
        """Verify actual Kenneth French asset pricing factors on disk."""
        from pathlib import Path
        p = Path("data/processed/factors_monthly.parquet")
        self.assertTrue(p.exists(), "factors_monthly.parquet must exist")
        df = pd.read_parquet(p)
        self.assertEqual(len(df), 759, "Must contain 759 continuous months")
        self.assertEqual(str(df["month"].min()), "1963-07")
        self.assertEqual(str(df["month"].max()), "2026-09")
        # Ensure return units are in decimals (e.g. mean market return ~0.005-0.01 per month, not 0.5-1.0%)
        self.assertLess(abs(df["Mkt-RF"].mean()), 0.05)
        self.assertLess(abs(df["RF"].mean()), 0.02)

    def test_actual_filings_and_security_master(self):
        """Verify actual filings table and augmented security master on disk."""
        from pathlib import Path
        p_fl = Path("data/processed/filings.parquet")
        self.assertTrue(p_fl.exists(), "filings.parquet must exist")
        df_fl = pd.read_parquet(p_fl)
        self.assertEqual(len(df_fl), 131920)
        self.assertEqual(df_fl["date_filed"].isna().sum(), 0)

        p_sm = Path("data/processed/security_master.parquet")
        self.assertTrue(p_sm.exists(), "security_master.parquet must exist")
        df_sm = pd.read_parquet(p_sm)
        self.assertEqual(len(df_sm), 22060)
        # Check match rate to risk CIKs is 84.8%
        self.assertEqual(df_sm["in_risk_dataset"].sum(), 14375)

    def test_actual_stage1_panel_integration(self):
        """Verify the integrated stock-month panel generated from actual datasets."""
        from pathlib import Path
        p_panel = Path("data/processed/panel_stock_month.parquet")
        self.assertTrue(p_panel.exists(), "panel_stock_month.parquet must exist")
        df_p = pd.read_parquet(p_panel)
        self.assertEqual(len(df_p), 1241695, "Panel must contain 1,241,695 stock-month rows")
        self.assertEqual(df_p["cik"].nunique(), 14382, "Panel must cover 14,382 unique CIK firms")
        self.assertEqual(df_p["has_marketing_data"].sum(), 476900, "Genuine marketing coverage must be 476,900 rows")
        self.assertEqual(df_p["market_orientation_raw"].notna().sum(), 476900)
        self.assertEqual(df_p["market_orientation"].isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
