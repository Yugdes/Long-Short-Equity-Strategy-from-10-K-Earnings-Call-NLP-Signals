"""
01_prepare_data.py
Loads the raw multi-label CSV, renames the text column, performs an
80-10-10 split, and writes three smaller CSVs into the repository’s
data folder. Uses a progress bar so that long reads show activity.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# Configuration comes from 00_config.py so that paths stay in one place.
# ----------------------------------------------------------------------
tqdm.pandas()  # add .progress_apply to Pandas
try:
    import code  # noqa: F401  (ensures relative import works during CLI run)
except ImportError:
    pass
from 00_config import (
    DATA_DIR,
    PROCESSED_TRAIN_DF,
)

# ----------------------------------------------------------------------
# 1. Locate the raw file
# ----------------------------------------------------------------------
if len(sys.argv) < 2:
    msg = "\nUsage: python 01_prepare_data.py <absolute_path_to_raw_csv>\n"
    raise SystemExit(msg)

raw_csv_path = Path(sys.argv[1]).expanduser().resolve()
if not raw_csv_path.is_file():
    raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

print(f"Reading raw data from: {raw_csv_path}")

# ----------------------------------------------------------------------
# 2. Load and clean
# ----------------------------------------------------------------------
df = pd.read_csv(raw_csv_path)
if "Example Sentence" not in df.columns:
    raise KeyError("'Example Sentence' column missing from raw file")

df = df.rename(columns={"Example Sentence": "text"})
print(f"Loaded {len(df):,} rows")

# ----------------------------------------------------------------------
# 3. Train/val/test split
# ----------------------------------------------------------------------
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)

# ----------------------------------------------------------------------
# 4. Write outputs
# ----------------------------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
train_path = DATA_DIR / "train.csv"
val_path   = DATA_DIR / "val.csv"
test_path  = DATA_DIR / "test.csv"

for sub_df, path in [(train_df, train_path),
                     (val_df,   val_path),
                     (test_df,  test_path)]:
    sub_df.to_csv(path, index=False)
    print(f"Wrote {len(sub_df):,} rows → {path}")

# also keep a single combined file for easy reload later
train_df.to_csv(PROCESSED_TRAIN_DF, index=False)
print(f"\nAll done. Combined training file saved to {PROCESSED_TRAIN_DF}")
