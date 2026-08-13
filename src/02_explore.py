"""
Exploratory analysis: what does the data look like, and which variables
appear related to claim frequency?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "policies_clean.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_PATH)
PORTFOLIO_FREQ = df["ClaimNb"].sum() / df["Exposure"].sum()

def frequency_by(data, column, bins=None):
    """
    Claim frequency within groups.

    Frequency is total claims divided by total EXPOSURE, not by policy
    count. A policy in force for one month had a twelfth of the chance
    to claim that a full-year policy had. Ignoring that makes short
    policies look artificially safe.
    """
    grouped = data.copy()
    if bins is not None:
        grouped["_band"] = pd.cut(grouped[column], bins=bins)
        key = "_band"
    else:
        key = column

    out = grouped.groupby(key, observed=True).agg(
        policies=("IDpol", "count"),
        claims=("ClaimNb", "sum"),
        exposure=("Exposure", "sum"),
    )
    out["frequency"] = out["claims"] / out["exposure"]
    out["vs_portfolio"] = out["frequency"] / PORTFOLIO_FREQ
    return out


def plot_frequency(table, title, filename):
    """
    Plot frequency on top, exposure volume below.

    The volume panel tells you which bars you are allowed to believe.
    A 40% frequency built on 30 policy-years is noise, not a finding.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    table["frequency"].plot(kind="bar", ax=ax1, color="steelblue")
    ax1.set_ylabel("Claims per year of exposure")
    ax1.set_title(title)
    ax1.axhline(PORTFOLIO_FREQ, color="red", linestyle="--",
                label="Portfolio average")
    ax1.legend()

    table["exposure"].plot(kind="bar", ax=ax2, color="lightgrey")
    ax2.set_ylabel("Exposure (policy-years)")
    ax2.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    print("=" * 62)
    print("SHAPE AND TYPES")
    print("=" * 62)
    print(f"Rows: {len(df):,} Columns: {len(df.columns)}")
    print(df.dtypes)
    print("\n" + "=" * 62)
    print("SUMMARY STATISTICS")
    print("=" * 62)
    print(df[["Exposure", "DrivAge", "VehAge",
              "BonusMalus", "Density"]].describe())
    
    print("\n" + "=" * 62)
    print("HEADLINE NUMBERS")
    print("=" * 62)
    print(f"Policies with at least one claim: {df['HasClaim'].mean():.2%}")
    print(f"Average exposure (years): {df['Exposure'].mean():.3f}")
    print(f"Exposure-adjusted frequency: {PORTFOLIO_FREQ:.4f} per year")

    print("\nCompare: naive rate ignoring exposure")
    print(f" Mean of HasClaim: {df['HasClaim'].mean():.4f}")
    print(f" True frequency: {PORTFOLIO_FREQ:.4f}")
    print(" These differ because average exposure is well under one year.")
    
    age = frequency_by(df, "DrivAge",
                      bins=[17, 21, 25, 30, 40, 50, 60, 70, 100])
    print("\n" + "=" * 62)
    print("FREQUENCY BY DRIVER AGE")
    print("=" * 62)
    print(age)
    plot_frequency(age, "Claim frequency by driver age",
                    "freq_by_drivage.png")
    
    bm = frequency_by(df, "BonusMalus",
                      bins=[49, 60, 80, 95, 100, 110, 130, 350])
    print("\nFREQUENCY BY BONUS-MALUS\n", bm)
    plot_frequency(bm, "Claim frequency by bonus-malus",
                  "freq_by_bonusmalus.png")
    
    area = frequency_by(df, "Area")
    print("\nFREQUENCY BY AREA\n", area)
    plot_frequency(area, "Claim frequency by area code",
                  "freq_by_area.png")
    
    veh = frequency_by(df, "VehAge", bins=[-1, 1, 3, 5, 10, 15, 100])
    print("\nFREQUENCY BY VEHICLE AGE\n", veh)
    plot_frequency(veh, "Claim frequency by vehicle age",
                  "freq_by_vehage.png")
    
    gas = frequency_by(df, "VehGas")
    print("\nFREQUENCY BY FUEL TYPE\n", gas)