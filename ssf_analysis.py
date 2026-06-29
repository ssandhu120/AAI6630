"""
Sit Stay Forever — CV Feature Analysis
Group A: Image Quality & Composition
Author: Nishant Chaudhari

Analysis layer: takes the computed CV features and answers the business question —
do higher-performing products have measurably different images, and where does SSF fall?

Run:  python ssf_analysis.py
Needs: pet_cv_dataset_full.xlsx in the same folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_FILE = "pet_cv_dataset_full.xlsx"

# CV features we analyze (numeric, computed per image)
CV_FEATURES = [
    "brightness_mean", "contrast_std", "sharpness_laplacian",
    "white_bg_pct", "white_bg_compliance", "product_dominance_score",
    "text_density_pct", "clutter_score", "color_warmth",
    "green_ratio", "saturation_mean", "symmetry_score",
    "dominant_color_1_pct", "ocr_word_count", "ocr_keyword_count",
]

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, sheet_name="images_cv_features")

# Keep only rows that actually have computed features (54 of 145)
df = df[df["brightness_mean"].notna()].copy()
df[CV_FEATURES] = df[CV_FEATURES].apply(pd.to_numeric, errors="coerce")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Images with computed CV features: {len(df)}")
print("\nBy performance tier:")
print(df["performance_tier"].value_counts().to_string())
print("\nBy image type:")
print(df["image_type_label"].value_counts().to_string())
print("\n** Note: 0 'low' tier images have features computed, and only 6 'medium'.")
print("   So we frame this as HIGH performers vs SPONSOR, not a 3-class problem. **")

# ── 2. FEATURE MEANS BY TIER ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MEAN CV FEATURES BY TIER")
print("=" * 70)
tier_means = df.groupby("performance_tier")[CV_FEATURES].mean().round(2)
print(tier_means.T.to_string())

# ── 3. HIGH vs SPONSOR GAP (the sponsor-relevant output) ──────────────────────
print("\n" + "=" * 70)
print("SSF GAP ANALYSIS — sponsor vs high performers")
print("=" * 70)
high = df[df["performance_tier"] == "high"]
ssf = df[df["performance_tier"] == "sponsor"]

gap = pd.DataFrame({
    "high_mean": high[CV_FEATURES].mean().round(2),
    "ssf_mean": ssf[CV_FEATURES].mean().round(2),
})
gap["ssf_minus_high"] = (gap["ssf_mean"] - gap["high_mean"]).round(2)
gap["pct_diff"] = ((gap["ssf_mean"] - gap["high_mean"]) / gap["high_mean"].abs() * 100).round(1)
gap = gap.sort_values("pct_diff", key=abs, ascending=False)
print(gap.to_string())
print("\nBiggest gaps (where SSF differs most from high performers):")
for feat in gap.head(4).index:
    direction = "below" if gap.loc[feat, "ssf_minus_high"] < 0 else "above"
    print(f"  - {feat}: SSF is {abs(gap.loc[feat,'pct_diff'])}% {direction} the high-performer average")

# ── 4. FEATURE IMPORTANCE (high vs rest) ──────────────────────────────────────
print("\n" + "=" * 70)
print("WHICH FEATURES DISTINGUISH HIGH PERFORMERS? (Random Forest)")
print("=" * 70)
# Binary target: is this image from a high-performing product?
data = df.dropna(subset=CV_FEATURES).copy()
data["is_high"] = (data["performance_tier"] == "high").astype(int)

X = data[CV_FEATURES]
y = data["is_high"]
print(f"Samples: {len(data)}  |  High: {y.sum()}  |  Not-high: {(y==0).sum()}")

rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X, y)

# Honest cross-validated accuracy (small data — interpret with caution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"5-fold CV accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
print("(Small/imbalanced dataset — the FEATURE RANKING matters more than accuracy.)")

importances = pd.Series(rf.feature_importances_, index=CV_FEATURES).sort_values(ascending=False)
print("\nFeature importance ranking:")
print(importances.round(3).to_string())

# ── 5. SAVE CHARTS ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

importances.head(8).iloc[::-1].plot.barh(ax=axes[0], color="#2E4D7B")
axes[0].set_title("Top CV features distinguishing high performers")
axes[0].set_xlabel("Importance")

top_gap = gap.head(8).iloc[::-1]
colors = ["#C77D3A" if v > 0 else "#7B2E2E" for v in top_gap["pct_diff"]]
axes[1].barh(top_gap.index, top_gap["pct_diff"], color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("SSF vs high performers (% difference)")
axes[1].set_xlabel("SSF relative to high-performer mean (%)")
axes[1].tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("ssf_analysis_charts.png", dpi=150, bbox_inches="tight")
print("\nSaved charts -> ssf_analysis_charts.png")

# Save the gap table for the report
gap.to_csv("ssf_gap_analysis.csv")
tier_means.T.to_csv("tier_means.csv")
print("Saved tables -> ssf_gap_analysis.csv, tier_means.csv")
