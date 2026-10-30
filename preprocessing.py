"""
Airline Passenger Satisfaction — Full Preprocessing Pipeline
=============================================================
Steps:
  1. Load & Inspect
  2. Drop irrelevant columns
  3. Handle missing values (Arrival Delay NaNs)
  4. Handle 0s in rating columns  (0 = "not applicable")
  5. Feature Engineering
       - Total Delay (Departure + Arrival), drop originals
       - Log-transform Flight Distance
       - Age group bins
       - Composite service scores (Ground & In-flight)
  6. Encode categorical features (Label Encoding for binary, OHE for multi-class)
  7. Scale numerical features (StandardScaler)
  8. Stratified Train / Validation / Test split  (70 / 15 / 15)
  9. Save processed artefacts
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE   = r"c:\Users\Krish\OneDrive\Desktop\ML_Project\ML_Project"
RAW    = os.path.join(BASE, "airline_passenger_satisfaction.csv")
OUT    = os.path.join(BASE, "processed")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(OUT,   exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n  STEP: {title}\n{SEP}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & Quick Inspect
# ─────────────────────────────────────────────────────────────────────────────
section("1 · Load & Inspect")
df = pd.read_csv(RAW)
print(f"Shape        : {df.shape}")
print(f"Columns      : {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nTarget distribution:\n{df['Satisfaction'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Drop Irrelevant Columns
# ─────────────────────────────────────────────────────────────────────────────
section("2 · Drop Irrelevant Columns")
df.drop(columns=["ID"], inplace=True)
print("Dropped: ID")
print(f"Shape after drop: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Handle Missing Values  (Arrival Delay: 393 NaNs)
# ─────────────────────────────────────────────────────────────────────────────
section("3 · Handle Missing Values")
print(f"Arrival Delay NaNs before : {df['Arrival Delay'].isnull().sum()}")

# Impute with MEDIAN (robust to delay outliers)
arrival_median = df["Arrival Delay"].median()
df["Arrival Delay"] = df["Arrival Delay"].fillna(arrival_median)

print(f"Arrival Delay NaNs after  : {df['Arrival Delay'].isnull().sum()}")
print(f"Imputed with median       : {arrival_median} minutes")
print(f"Total NaNs remaining      : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Handle 0s in Rating Columns  (0 = "Not Applicable", NOT a rating)
# ─────────────────────────────────────────────────────────────────────────────
section("4 · Handle 0s in Rating Columns  (0 = Not Applicable)")

RATING_COLS = [
    "Departure and Arrival Time Convenience",
    "Ease of Online Booking",
    "Check-in Service",
    "Online Boarding",
    "Gate Location",
    "On-board Service",
    "Seat Comfort",
    "Leg Room Service",
    "Cleanliness",
    "Food and Drink",
    "In-flight Service",
    "In-flight Wifi Service",
    "In-flight Entertainment",
    "Baggage Handling",
]

print("\n0-counts per rating column BEFORE imputation:")
for col in RATING_COLS:
    n = (df[col] == 0).sum()
    if n > 0:
        print(f"  {col:<45} {n:>6}")

# Strategy: Replace 0 → NaN, then impute with column MEDIAN (1–5 scale)
# This preserves the ordinal meaning without creating a false "zero" rating
for col in RATING_COLS:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, np.nan).fillna(median_val)

print("\n0-counts per rating column AFTER imputation:")
for col in RATING_COLS:
    print(f"  {col:<45} {(df[col] == 0).sum():>6}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
section("5 · Feature Engineering")

# 5a — Total Delay  (Departure + Arrival are 0.965 correlated — merge them)
df["Total Delay"] = df["Departure Delay"] + df["Arrival Delay"]
df.drop(columns=["Departure Delay", "Arrival Delay"], inplace=True)
print("✔  Created 'Total Delay' = Departure Delay + Arrival Delay")
print("   Dropped 'Departure Delay' and 'Arrival Delay'")

# 5b — Log-transform Flight Distance  (right-skewed: max 4983, mean 1190)
df["Log Flight Distance"] = np.log1p(df["Flight Distance"])
df.drop(columns=["Flight Distance"], inplace=True)
print("✔  Created 'Log Flight Distance' = log1p(Flight Distance)")
print("   Dropped 'Flight Distance'")

# 5c — Age Group Bins
age_bins   = [0,  17,  25,  40,  60,  100]
age_labels = ["Child", "Young Adult", "Adult", "Middle-Aged", "Senior"]
df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
print("✔  Created 'Age Group' with bins:", dict(zip(age_labels, age_bins)))

# 5d — Composite Service Scores
ground_cols   = ["Departure and Arrival Time Convenience",
                 "Ease of Online Booking",
                 "Check-in Service",
                 "Gate Location"]

inflight_cols = ["Online Boarding",
                 "On-board Service",
                 "Seat Comfort",
                 "Leg Room Service",
                 "Cleanliness",
                 "Food and Drink",
                 "In-flight Service",
                 "In-flight Wifi Service",
                 "In-flight Entertainment",
                 "Baggage Handling"]

df["Ground Service Score"]    = df[ground_cols].mean(axis=1).round(2)
df["In-flight Service Score"] = df[inflight_cols].mean(axis=1).round(2)
print(f"✔  Created 'Ground Service Score'    (mean of {len(ground_cols)} features)")
print(f"✔  Created 'In-flight Service Score' (mean of {len(inflight_cols)} features)")

print(f"\nShape after Feature Engineering: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Encode Categorical Features
# ─────────────────────────────────────────────────────────────────────────────
section("6 · Encode Categorical Features")

# ── 6a. Binary columns → Label Encode (0/1)
binary_map = {
    "Gender"        : {"Male": 0, "Female": 1},
    "Customer Type" : {"First-time": 0, "Returning": 1},
    "Type of Travel": {"Personal": 0, "Business": 1},
    "Satisfaction"  : {"Neutral or Dissatisfied": 0, "Satisfied": 1},
}
for col, mapping in binary_map.items():
    df[col] = df[col].map(mapping)
    print(f"  Label Encoded: {col}  →  {mapping}")

# ── 6b. Class (3 levels) → One-Hot Encode
df = pd.get_dummies(df, columns=["Class"], prefix="Class", drop_first=False)
print("\n  One-Hot Encoded: 'Class'  →  Class_Business, Class_Economy, Class_Economy Plus")

# ── 6c. Age Group (ordinal but treat as OHE for simplicity)
df = pd.get_dummies(df, columns=["Age Group"], prefix="AgeGrp", drop_first=False)
print("  One-Hot Encoded: 'Age Group'  →  AgeGrp_Child, AgeGrp_Young Adult, ...")

# Drop Age (replaced by Age Group dummies)
df.drop(columns=["Age"], inplace=True)
print("  Dropped raw 'Age' column (replaced by 'Age Group' dummies)")

print(f"\nShape after Encoding: {df.shape}")
print(f"Columns:\n{df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Scale Numerical Features
# ─────────────────────────────────────────────────────────────────────────────
section("7 · Scale Numerical Features  (StandardScaler)")

# Columns to scale: continuous numerics only (not encoded 0/1 or ordinal ratings)
SCALE_COLS = [
    "Total Delay",
    "Log Flight Distance",
    "Ground Service Score",
    "In-flight Service Score",
] + RATING_COLS   # individual rating cols also kept

# Only scale columns that still exist
SCALE_COLS = [c for c in SCALE_COLS if c in df.columns]
print(f"Scaling {len(SCALE_COLS)} columns: {SCALE_COLS}")

scaler = StandardScaler()
df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
print("✔  StandardScaler applied")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Stratified Train / Validation / Test Split  (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────────────────────
section("8 · Stratified Train / Val / Test Split  (70 / 15 / 15)")

X = df.drop(columns=["Satisfaction"])
y = df["Satisfaction"]

# First split → 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Second split → 50% of temp = 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training set   : {X_train.shape[0]:>7,} rows  ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation set : {X_val.shape[0]:>7,} rows  ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set       : {X_test.shape[0]:>7,} rows  ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\nTarget class balance check (stratified):")
for name, s in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    pct = s.value_counts(normalize=True).mul(100).round(1)
    print(f"  {name}: Satisfied={pct.get(1,0)}%  Dissatisfied={pct.get(0,0)}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save Processed Artefacts
# ─────────────────────────────────────────────────────────────────────────────
section("9 · Save Processed Artefacts")

X_train.to_csv(os.path.join(OUT, "X_train.csv"), index=False)
X_val.to_csv(  os.path.join(OUT, "X_val.csv"),   index=False)
X_test.to_csv( os.path.join(OUT, "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(OUT, "y_train.csv"), index=False)
y_val.to_csv(  os.path.join(OUT, "y_val.csv"),   index=False)
y_test.to_csv( os.path.join(OUT, "y_test.csv"),  index=False)

# Also save the full processed dataframe
df.to_csv(os.path.join(OUT, "processed_full.csv"), index=False)

# Save scaler column info for later use
import json
scaler_info = {
    "scale_cols"    : SCALE_COLS,
    "scaler_mean_"  : scaler.mean_.tolist(),
    "scaler_scale_" : scaler.scale_.tolist(),
}
with open(os.path.join(OUT, "scaler_info.json"), "w") as f:
    json.dump(scaler_info, f, indent=2)

print(f"Saved to: {OUT}")
print("  X_train.csv, X_val.csv, X_test.csv")
print("  y_train.csv, y_val.csv, y_test.csv")
print("  processed_full.csv")
print("  scaler_info.json")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Quick Sanity Plots
# ─────────────────────────────────────────────────────────────────────────────
section("10 · Generating Sanity Plots")

# Plot 1 – Target distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Airline Passenger Satisfaction — Post-Preprocessing Sanity Check", fontsize=13)

# Class balance
y_counts = y_train.value_counts()
axes[0].bar(["Dissatisfied (0)", "Satisfied (1)"], y_counts.values, color=["#e74c3c","#2ecc71"])
axes[0].set_title("Target Distribution (Train)")
axes[0].set_ylabel("Count")
for i, v in enumerate(y_counts.values):
    axes[0].text(i, v + 200, str(v), ha="center", fontweight="bold")

# Total Delay distribution
axes[1].hist(X_train["Total Delay"], bins=60, color="#3498db", edgecolor="white", alpha=0.8)
axes[1].set_title("Total Delay (Scaled)")
axes[1].set_xlabel("Scaled Value")
axes[1].set_ylabel("Count")

# Log Flight Distance distribution
axes[2].hist(X_train["Log Flight Distance"], bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[2].set_title("Log Flight Distance (Scaled)")
axes[2].set_xlabel("Scaled Value")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_sanity_check.png"), dpi=120, bbox_inches="tight")
plt.close()
print("✔  Saved: plots/01_sanity_check.png")

# Plot 2 – Feature correlation heatmap (numeric cols only)
num_cols_for_corr = [c for c in X_train.columns
                     if X_train[c].dtype in [np.float64, np.int64, float, int]
                     and not c.startswith("Class_")
                     and not c.startswith("AgeGrp_")][:15]

fig, ax = plt.subplots(figsize=(12, 9))
corr = X_train[num_cols_for_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap (Training Set)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_correlation_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("✔  Saved: plots/02_correlation_heatmap.png")

print(f"\n{'='*65}")
print("  ✅  PREPROCESSING COMPLETE")
print(f"  Final feature count : {X_train.shape[1]}")
print(f"  Train rows          : {X_train.shape[0]:,}")
print(f"  Val rows            : {X_val.shape[0]:,}")
print(f"  Test rows           : {X_test.shape[0]:,}")
print(f"{'='*65}\n")
