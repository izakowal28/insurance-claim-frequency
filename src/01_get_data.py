"""
Download the French MTPL dataset, clean it, and save analysis-ready files.
Run this first. Everything downstream depends on its output.
"""
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
# Paths are built relative to THIS FILE, not to your username's home folder.
# This is what makes the project run on someone else's computer unchanged.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "policies_clean.csv"
EXCEL_PATH = PROJECT_ROOT / "excel" / "policies_sample.xlsx"

SOURCE_URL = "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff"

# exist_ok=True means "do not crash if the folder is already there"
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def download_if_missing():
    """Fetch the raw file only if it is not already on disk."""
    if RAW_PATH.exists():
        print(f"Raw file already present: {RAW_PATH}")
        return
    print("Downloading from OpenML. This may take a minute...")
    urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)
    print(f"Saved to {RAW_PATH}")

def load_raw():
    """Read the CSV and repair formatting damage from the ARFF conversion."""
    df = pd.read_csv(RAW_PATH)

    # The conversion sometimes leaves single quotes around text values and
    # column names. Strip them, or 'D' and D become two different area codes.
    df.columns = df.columns.str.strip().str.strip("'\"")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip("'\"")

    # Force numeric columns to actually be numeric. If quoting turned them
    # into text, this repairs it. errors="coerce" converts anything
    # unparseable into a missing value instead of crashing, so we can count
    # the damage afterwards rather than guessing at it.
    for col in ["IDpol", "ClaimNb", "Exposure", "VehPower",
                "VehAge", "DrivAge", "BonusMalus", "Density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def clean(df):
    """Apply documented data-quality fixes and build derived columns."""
    n_start = len(df)

    # Exposure is the fraction of a year the policy was in force. Zero or
    # negative exposure is meaningless and would cause division by zero
    # in every frequency calculation downstream.
    df = df[df["Exposure"] > 0].copy()

    # This dataset has known, documented quality problems: a small number of
    # records carry impossible claim counts, and some exposures exceed one
    # year. Capping is the standard treatment in the actuarial literature.
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    # THE TARGET VARIABLE for the logistic model: did this policy produce
    # at least one claim? astype(int) turns True/False into 1/0.
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

    # Density is inhabitants per square kilometre and is heavily
    # right-skewed: most areas are sparse, a few are enormously dense.
    # A log transform compresses that tail so a linear model can use it.
    # log1p computes log(1 + x), which stays safe if any value is zero.
    df["LogDensity"] = np.log1p(df["Density"])

    print(f"Rows before cleaning: {n_start:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Duplicate policy IDs: {df['IDpol'].duplicated().sum():,}")
    print(f"Missing values total: {df.isna().sum().sum():,}")

    return df
if __name__ == "__main__":
    download_if_missing()
    df = load_raw()
    df = clean(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned data: {CLEAN_PATH}")

    # Excel struggles with 678,000 rows. Save a reproducible random sample.
    sample = df.sample(n=20000, random_state=42)
    sample.to_excel(EXCEL_PATH, index=False)
    print(f"Saved Excel sample: {EXCEL_PATH}")

    # SANITY CHECK: the sample should show roughly the same claim frequency
    # as the full dataset. If it does not, the sampling is broken.
    full_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
    samp_freq = sample["ClaimNb"].sum() / sample["Exposure"].sum()
    print(f"\nFull data claim frequency: {full_freq:.4f}")
    print(f"Sample claim frequency: {samp_freq:.4f}")