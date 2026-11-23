"""Kenneth French data library: FF5 (2x3) + momentum + RF (monthly, decimals) and SIC->FF48/FF12 maps."""
import io
import re
import zipfile

import pandas as pd
import requests

from src.utils.config import path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip", "mom": "F-F_Momentum_Factor_CSV.zip",
         "sic48": "Siccodes48.zip", "sic12": "Siccodes12.zip", "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip"}


def _fetch(key: str) -> bytes:
    d = path("raw") / "french"
    d.mkdir(parents=True, exist_ok=True)
    f = d / FILES[key]
    if not f.exists():
        r = requests.get(BASE + FILES[key], timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    with zipfile.ZipFile(f) as z:
        return z.read(z.namelist()[0])


def _monthly_block(text: str) -> pd.DataFrame:
    """First block of YYYYMM rows (the monthly table) in a French CSV."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", l)) - 1
    rows = []
    for l in lines[hdr + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", l):
            break
        rows.append([v.strip() for v in l.split(",")])
    cols = ["yyyymm"] + [c.strip() for c in lines[hdr].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df["month"] = pd.PeriodIndex(df.pop("yyyymm"), freq="M")
    return df.set_index("month").astype(float) / 100.0


def load_ff_factors() -> pd.DataFrame:
    out = path("interim") / "factors.parquet"
    if out.exists():
        return pd.read_parquet(out)
    ff5 = _monthly_block(_fetch("ff5").decode("latin-1"))
    mom = _monthly_block(_fetch("mom").decode("latin-1"))
    mom.columns = ["Mom"]
    f = ff5.join(mom, how="left").rename(columns=lambda c: c.replace("-", "").lower())
    f = f.rename(columns={"mktrf": "mktrf"}).reset_index()
    f.to_parquet(out, index=False)
    return f


def load_ff_daily() -> pd.DataFrame:
    """Daily FF3 + RF (decimals): [date, mktrf, smb, hml, rf]. Used only by the filing-date event study."""
    text = _fetch("ff3_daily").decode("latin-1").splitlines()
    hdr = next(i for i, l in enumerate(text) if re.match(r"^\s*\d{8}\s*,", l)) - 1
    rows = [[v.strip() for v in l.split(",")] for l in text[hdr + 1:] if re.match(r"^\s*\d{8}\s*,", l)]
    df = pd.DataFrame(rows, columns=["date"] + [c.strip().replace("-", "").lower() for c in text[hdr].split(",")[1:]])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float).div(100).reset_index()


def load_sic_map(scheme: str = "sic48") -> pd.DataFrame:
    """[lo, hi, ind, ind_name] ranges from Siccodes48/12. SICs outside all ranges are 'Other'."""
    text = _fetch(scheme).decode("latin-1")
    rows, cur = [], None
    for l in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s+(\S+)\s+(.*)$", l)
        r = re.match(r"^\s*(\d{4})-(\d{4})", l)
        if m and not r:
            cur = (int(m.group(1)), m.group(2))
        elif r and cur:
            rows.append((int(r.group(1)), int(r.group(2)), cur[0], cur[1]))
    return pd.DataFrame(rows, columns=["lo", "hi", "ind", "ind_name"])


def sic_to_industry(sic: pd.Series, scheme: str = "sic48") -> pd.Series:
    mp = load_sic_map(scheme)
    other = 48 if scheme == "sic48" else 12
    out = pd.Series(other, index=sic.index, dtype="int64")
    for lo, hi, ind, _ in mp.itertuples(index=False):
        out[(sic >= lo) & (sic <= hi)] = ind
    out[sic.isna()] = -1
    return out
