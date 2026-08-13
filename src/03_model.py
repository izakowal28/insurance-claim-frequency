"""
Two models of the same question.

1. Logistic regression: will this customer file a claim, yes or no?
2. Poisson GLM: how many claims per year of exposure will they generate?

The second is what a pricing actuary actually builds, because it uses the
claim count and corrects for how long each policy was observed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "policies_clean.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_PATH)
PORTFOLIO_FREQ = df["ClaimNb"].sum() / df["Exposure"].sum()

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
FEATURES = NUMERIC + CATEGORICAL

X = df[FEATURES]
y = df["HasClaim"]

# stratify=y keeps the same claim rate in train and test. Without it,
# random chance could hand you a test set with a different base rate and
# your metrics would be measuring the split, not the model.
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
X, y, df.index, test_size=0.25, random_state=42, stratify=y
)

print(f"Training rows: {len(X_train):,} Test rows: {len(X_test):,}")
print(f"Claim rate in train: {y_train.mean():.4f}")
print(f"Claim rate in test: {y_test.mean():.4f}")

# =====================================================================
# MODEL 1: LOGISTIC REGRESSION
# =====================================================================

# A Pipeline bundles preprocessing and the model into one object.
# This matters: the scaler learns its mean and standard deviation from
# the TRAINING data only. Scaling everything before splitting would leak
# information from the test set into training and your score would be a lie.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
])

logit = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=2000)),
])

print("\nFitting logistic regression...")
logit.fit(X_train, y_train)

probs = logit.predict_proba(X_test)[:, 1]

# =====================================================================
# WHY ACCURACY IS THE WRONG METRIC
# =====================================================================
print("\n" + "=" * 62)
print("WHY ACCURACY IS THE WRONG METRIC")
print("=" * 62)
always_no = np.zeros_like(y_test)
print(f"Accuracy, model that always says NO: {accuracy_score(y_test, always_no):.4f}")
print(f"Accuracy, our model at threshold .50: "
    f"{accuracy_score(y_test, (probs >= 0.50).astype(int)):.4f}")
print("Nearly identical. A useless model scores the same as a real one,")
print("because about 95% of policies genuinely produce no claim.")

# =====================================================================
# METRICS THAT MEASURE SOMETHING
# =====================================================================
auc = roc_auc_score(y_test, probs)
gini = 2 * auc - 1

print("\n" + "=" * 62)
print("METRICS THAT ACTUALLY MEASURE SOMETHING")
print("=" * 62)
print(f"ROC-AUC: {auc:.4f} (0.50 = coin flip, 1.00 = perfect)")
print(f"Gini: {gini:.4f} (= 2*AUC - 1, the insurance standard)")

# At a lower threshold the model actually flags people, so precision
# and recall become meaningful numbers rather than degenerate ones.
threshold = np.percentile(probs, 90)
flagged = (probs >= threshold).astype(int)

print(f"\nAt a threshold flagging the riskiest 10% of policies:")
print(f" Precision: {precision_score(y_test, flagged):.4f}"
      f" (of those flagged, share that actually claimed)")
print(f" Recall: {recall_score(y_test, flagged):.4f}"
      f" (of all claimers, share we caught)")

cm = confusion_matrix(y_test, flagged)
print("\nConfusion matrix (rows = actual, cols = predicted):")
print(f" Predicted no Predicted yes")
print(f" Actual no {cm[0, 0]:>10,} {cm[0, 1]:>13,}")
print(f" Actual yes {cm[1, 0]:>10,} {cm[1, 1]:>13,}")

# =====================================================================
# ROC CURVE
# =====================================================================
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, linewidth=2, label=f"Logistic regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1],"k--", label="Random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve: claim vs no claim")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
plt.close()
print("\nSaved roc_curve.png")

# =====================================================================
# LIFT CHART: the chart that matters commercially
# =====================================================================
lift_df = pd.DataFrame({
    "prob": probs,
    "claims": df.loc[idx_test, "ClaimNb"].values,
    "exposure": df.loc[idx_test, "Exposure"].values,
})
lift_df["decile"] = pd.qcut(lift_df["prob"], 10, labels=False,
                            duplicates="drop")

lift = lift_df.groupby("decile").agg(
    claims=("claims", "sum"),
    exposure=("exposure", "sum"),
)
lift["frequency"] = lift["claims"] / lift["exposure"]
lift["vs_portfolio"] = lift["frequency"] / PORTFOLIO_FREQ

print("\n" + "=" * 62)
print("LIFT: ACTUAL FREQUENCY BY PREDICTED-RISK DECILE")
print("=" * 62)
print(lift)

ratio = lift["frequency"].iloc[-1] / lift["frequency"].iloc[0]
print(f"\nRiskiest decile vs safest decile: {ratio:.2f}x")
print("This is your headline business number.")

plt.figure(figsize=(9, 6))
plt.bar(lift.index, lift["frequency"], color="steelblue")
plt.axhline(PORTFOLIO_FREQ, color="red", linestyle="--",
            label="Portfolio average")
plt.xlabel("Predicted risk decile (0 = safest, 9 = riskiest)")
plt.ylabel("Actual claim frequency")
plt.title("Model lift: does the ranking hold on unseen data?")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "lift_chart.png", dpi=150)
plt.close()
print("Saved lift_chart.png")

# =====================================================================
# MODEL 2: POISSON GLM WITH EXPOSURE OFFSET
# =====================================================================
print("\n" + "=" * 62)
print("POISSON FREQUENCY GLM")
print("=" * 62)

train_df = df.loc[idx_train]

X_glm = pd.get_dummies(train_df[FEATURES], drop_first=True).astype(float)
X_glm = sm.add_constant(X_glm)

# The offset is a term whose coefficient is FIXED at 1. It encodes the
# fact we already know: twice the exposure means twice the expected
# claims. Letting the model estimate that relationship would mean
# fitting noise to something we know from first principles.
poisson = sm.GLM(
    train_df["ClaimNb"],
    X_glm,
    family=sm.families.Poisson(),
    offset=np.log(train_df["Exposure"]),
).fit()

print(poisson.summary())

# exp(coefficient) is the multiplicative effect on claim frequency.
# In pricing language, that number IS the rate relativity.
relativities = pd.DataFrame({
    "coefficient": poisson.params,
    "relativity": np.exp(poisson.params),
    "p_value": poisson.pvalues,
}).sort_values("relativity", ascending=False)

print("\n" + "=" * 62)
print("RATE RELATIVITIES: strongest upward effects")
print("=" * 62)
print(relativities.head(12))
print("\nRATE RELATIVITIES: strongest downward effects")
print(relativities.tail(8))
out_path = PROJECT_ROOT / "outputs" / "relativities.csv"
relativities.to_csv(out_path)
print(f"\nSaved {out_path}")