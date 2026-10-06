"""
Generates dummy data for the NLP risk scores and marketing emphasis
so that the pipeline can be tested end-to-end without downloading the real datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_dummy_data():
    np.random.seed(42)
    print("Generating dummy data for testing...")
    
    # Ensure directories exist
    Path("data/raw/marketing").mkdir(parents=True, exist_ok=True)
    
    # 1. Dummy Risk Scores (data set.xlsx)
    # 50 companies over 10 years
    ciks = np.random.choice(range(10000, 999999), 50, replace=False)
    years = list(range(2010, 2021))
    
    risk_rows = []
    for cik in ciks:
        for year in years:
            row = {
                "cik": cik,
                "year": year,
                "item_1a_optional": 0,
                "item_1a_word_count": np.random.randint(500, 5000)
            }
            # Add 8 factors
            factors = ["accounting", "finance", "international", "legal",
                       "management", "marketing", "operations", "technology"]
            for f in factors:
                row[f"{f}_prob"] = np.random.uniform(0, 1)
                row[f"{f}_count"] = np.random.randint(0, 50)
                row[f"{f}_binary"] = 1 if row[f"{f}_prob"] > 0.5 else 0
            risk_rows.append(row)
            
    risk_df = pd.DataFrame(risk_rows)
    # Save to Excel
    excel_path = "data/raw/data set.xlsx"
    risk_df.to_excel(excel_path, index=False)
    print(f"Created dummy risk data at {excel_path} ({len(risk_df)} rows)")
    
    # 2. Dummy Marketing Scores (CSV)
    mktg_rows = []
    for cik in ciks:
        for year in years:
            for qtr in range(1, 5):
                row = {
                    "cik": cik, # using CIK for easy merging in the dummy
                    "year": year,
                    "quarter": qtr,
                    "market_orientation": np.random.normal(0, 1),
                    "marketing_capabilities": np.random.normal(0, 1),
                    "marketing_excellence": np.random.normal(0, 1)
                }
                mktg_rows.append(row)
                
    mktg_df = pd.DataFrame(mktg_rows)
    csv_path = "data/raw/marketing/dummy_marketing.csv"
    mktg_df.to_csv(csv_path, index=False)
    print(f"Created dummy marketing data at {csv_path} ({len(mktg_df)} rows)")
    
    print("\nDummy data generation complete! You can now run `make all`")

if __name__ == "__main__":
    generate_dummy_data()
