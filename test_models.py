"""
Airline Passenger Satisfaction — Model Testing & Single Point Prediction
========================================================================
This script:
  1. Loads all 3 saved models (LightGBM, XGBoost, Random Forest) from `saved_models/`
  2. Evaluates and displays the complete test metrics (Accuracy, F1, ROC-AUC, Precision, Recall)
  3. Applies test points (sample passengers) across all 3 models to predict satisfaction in real-time
"""

import os
import sys
import io
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── File Paths ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "saved_models")
PROC_DIR   = os.path.join(BASE, "processed")

# ─── Load Test Data ───────────────────────────────────────────────────────────
print("=" * 70)
print("1. LOADING HELD-OUT TEST DATA")
print("=" * 70)
X_test = pd.read_csv(os.path.join(PROC_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC_DIR, "y_test.csv")).squeeze()
print(f"Loaded {len(X_test):,} unseen passenger records with {X_test.shape[1]} features each.\n")

# ─── Load Saved Models ────────────────────────────────────────────────────────
print("=" * 70)
print("2. LOADING SAVED .PKL MODELS FROM DISK")
print("=" * 70)
models = {
    "LightGBM": joblib.load(os.path.join(MODELS_DIR, "lightgbm.pkl")),
    "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl")),
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
}
for name in models:
    file_name = f"{name.lower().replace(' ', '_')}.pkl"
    print(f"[*] Successfully loaded {name} from saved_models/{file_name}")
print()

# ─── Overall Metrics on Full Test Set ─────────────────────────────────────────
print("=" * 70)
print("3. OVERALL EVALUATION METRICS ON TEST DATA (19,482 PASSENGERS)")
print("=" * 70)
metrics_list = []

for name, model in models.items():
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    metrics_list.append({
        "Model": name,
        "Accuracy": f"{acc * 100:.2f}%",
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Precision": f"{prec * 100:.2f}%",
        "Recall": f"{rec * 100:.2f}%",
        "TN": f"{tn:,}",
        "FP": f"{fp:,}",
        "FN": f"{fn:,}",
        "TP": f"{tp:,}"
    })

df_metrics = pd.DataFrame(metrics_list)
print(df_metrics.to_string(index=False))
print("\n")

# ─── Single Test Point Inference ──────────────────────────────────────────────
print("=" * 70)
print("4. APPLYING INDIVIDUAL TEST POINTS ACROSS ALL 3 MODELS")
print("=" * 70)

# We test 2 contrasting real passenger profiles from the test set:
# Passenger A: Business Class traveler who had a great flight
# Passenger B: Economy Class traveler with lower survey ratings

test_indices = [1, 0]  # Index 1 = Satisfied passenger, Index 0 = Dissatisfied passenger

for count, idx in enumerate(test_indices, start=1):
    test_passenger = X_test.iloc[[idx]]
    actual_label   = "Satisfied (1)" if y_test.iloc[idx] == 1 else "Neutral / Dissatisfied (0)"
    
    # Key passenger highlights for display
    is_business = "Business" if test_passenger["Class_Business"].values[0] == 1 else "Economy"
    travel_type = "Business Travel" if test_passenger["Type of Travel"].values[0] == 1 else "Personal Travel"
    
    print(f"\n--- [TEST POINT #{count}] Passenger #{idx + 1} Profile ---")
    print(f" • Travel Class: {is_business} | Purpose: {travel_type}")
    print(f" • Ground Truth (Actual Experience): {actual_label}")
    print("-" * 62)
    print(f"{'Model':<16} | {'Predicted Label':<26} | {'Confidence Probability'}")
    print("-" * 62)
    
    for name, model in models.items():
        pred = model.predict(test_passenger)[0]
        prob = model.predict_proba(test_passenger)[0][1] # Probability of being Satisfied
        
        pred_label = "Satisfied" if pred == 1 else "Neutral or Dissatisfied"
        confidence = prob * 100 if pred == 1 else (1 - prob) * 100
        
        match_icon = "[CORRECT]" if (pred == y_test.iloc[idx]) else "[WRONG]"
        print(f"{name:<16} | {pred_label:<24} | {confidence:.2f}% {match_icon}")

print("\n" + "=" * 70)
print("TEST SCRIPT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)
