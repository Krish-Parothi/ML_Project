"""
Airline Passenger Satisfaction — Streamlined High-Performance Training Pipeline
================================================================================
Models Kept (The Top 3 Best Performers):
  1. LightGBM      (Best overall: AUC ~0.994, Acc ~96.0%)
  2. XGBoost       (Runner-up: AUC ~0.993, Acc ~95.8%)
  3. Random Forest (Top bagging ensemble: AUC ~0.992, Acc ~95.5%)

All slow, low-performing, and redundant models (Logistic Regression, SVM,
KNN, AdaBoost, Decision Tree, GradientBoosting) have been removed.

Pipeline:
  1. Train models (using best fine-tuned hyperparameters)
  2. Evaluate on Validation & held-out Test sets
  3. Save trained models to `saved_models/` (.pkl)
  4. Generate and save evaluation plots & metrics CSV
"""

import os, sys, io, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
import xgboost as xgb
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

def flush(): sys.stdout.flush()
SEP = "=" * 68
def section(t): print(f"\n{SEP}\n  {t}\n{SEP}"); flush()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = r"c:\Users\Krish\OneDrive\Desktop\ML_Project\ML_Project"
PROC       = os.path.join(BASE, "processed")
PLOTS      = os.path.join(BASE, "plots")
RESULTS    = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "saved_models")

for d in [RESULTS, PLOTS, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
section("1. LOADING PROCESSED DATA")
X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
X_val   = pd.read_csv(os.path.join(PROC, "X_val.csv"))
X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv")).squeeze()
y_val   = pd.read_csv(os.path.join(PROC, "y_val.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv")).squeeze()

print(f"X_train : {X_train.shape}  |  X_val : {X_val.shape}  |  X_test : {X_test.shape}")
print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")
flush()

# ─── Metric Helper ────────────────────────────────────────────────────────────
def evaluate(name, model, X, y, split="Test"):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds)
    prec  = precision_score(y, preds)
    rec   = recall_score(y, preds)
    auc   = roc_auc_score(y, proba) if proba is not None else np.nan
    print(f"  [{split}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    flush()
    return {
        "Model": name,
        "Split": split,
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4)
    }

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS WITH OPTIMIZED / BEST HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
section("2. DEFINING TOP-TIER MODELS WITH OPTIMIZED HYPERPARAMETERS")

MODELS = {
    # 1. LightGBM (Tuned Best Hyperparameters)
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.5,
        reg_lambda=0.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ),

    # 2. XGBoost (Tuned Best Hyperparameters)
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ),

    # 3. Random Forest (Optimized Ensemble Bagging)
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
section("3. TRAINING & EVALUATING MODELS")

all_evaluations = []
trained_models = {}

for name, model in MODELS.items():
    print(f"\n>>> Training: {name}...")
    flush()
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training Time: {elapsed:.2f} seconds")
    flush()

    # Evaluate on Train, Validation, and Test
    evaluate(name, model, X_train, y_train, "Train")
    val_res  = evaluate(name, model, X_val,   y_val,   "Val")
    test_res = evaluate(name, model, X_test,  y_test,  "Test")
    
    all_evaluations.extend([val_res, test_res])
    trained_models[name] = model

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS (.pkl)
# ═══════════════════════════════════════════════════════════════════════════════
section("4. SAVING TRAINED MODELS")

for name, model in trained_models.items():
    file_name = name.lower().replace(" ", "_") + ".pkl"
    file_path = os.path.join(MODELS_DIR, file_name)
    joblib.dump(model, file_path)
    print(f"  ✔ Saved model: saved_models/{file_name}")
    flush()

# Save the overall best model (LightGBM) as 'best_model.pkl'
best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(trained_models["LightGBM"], best_model_path)
print(f"  ✔ Saved primary production model: saved_models/best_model.pkl")
flush()

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY & CSV
# ═══════════════════════════════════════════════════════════════════════════════
section("5. LEADERBOARD & RESULTS")

df_results = pd.DataFrame(all_evaluations)
df_test = df_results[df_results["Split"] == "Test"].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

csv_path = os.path.join(RESULTS, "final_model_performance.csv")
df_results.to_csv(csv_path, index=False)
print(f"  ✔ Results saved to: results/final_model_performance.csv\n")
print(df_test[["Model", "Accuracy", "F1", "ROC-AUC", "Precision", "Recall"]].to_string(index=False))
flush()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("6. GENERATING EVALUATION PLOTS")

# Plot 1: Performance Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Final Model Comparison on Held-Out Test Set", fontsize=14, fontweight="bold")
palette = ["#2ecc71", "#3498db", "#9b59b6"]

for ax, metric in zip(axes, ["Accuracy", "F1", "ROC-AUC"]):
    bars = ax.bar(df_test["Model"], df_test[metric], color=palette, width=0.55)
    ax.set_ylim(0.90, 1.0)
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel("Score")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
p1_path = os.path.join(PLOTS, "final_model_comparison.png")
plt.savefig(p1_path, dpi=130)
plt.close()
print("  ✔ Saved: plots/final_model_comparison.png")

# Plot 2: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices on Held-Out Test Set", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes, trained_models.items()):
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=["Dissatisfied", "Satisfied"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc   = accuracy_score(y_test, preds)
    ax.set_title(f"{name}\nAccuracy: {acc:.4f}", fontsize=11)

plt.tight_layout()
p2_path = os.path.join(PLOTS, "final_confusion_matrices.png")
plt.savefig(p2_path, dpi=130)
plt.close()
print("  ✔ Saved: plots/final_confusion_matrices.png")

# Plot 3: ROC Curves
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"LightGBM": "#2ecc71", "XGBoost": "#3498db", "Random Forest": "#9b59b6"}

for name, model in trained_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    RocCurveDisplay.from_predictions(
        y_test, proba, name=f"{name} (AUC = {auc:.4f})",
        ax=ax, color=colors[name], lw=2
    )

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (0.5000)")
ax.set_title("ROC Curves Comparison (Test Set)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
p3_path = os.path.join(PLOTS, "final_roc_curves.png")
plt.savefig(p3_path, dpi=130)
plt.close()
print("  ✔ Saved: plots/final_roc_curves.png")

# Plot 4: Top 20 Feature Importance (from LightGBM)
lgb_model = trained_models["LightGBM"]
fi = pd.Series(lgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 7))
fi.plot(kind="barh", ax=ax, color="#1abc9c")
ax.set_title("Top 20 Most Important Features (LightGBM)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
p4_path = os.path.join(PLOTS, "top_feature_importance.png")
plt.savefig(p4_path, dpi=130)
plt.close()
print("  ✔ Saved: plots/top_feature_importance.png")

# ─── Detailed Classification Report ───────────────────────────────────────────
section("7. DETAILED CLASSIFICATION REPORT (BEST MODEL: LightGBM)")
y_pred_best = trained_models["LightGBM"].predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["Neutral or Dissatisfied", "Satisfied"]))

section("ALL COMPLETE! Pipeline ran successfully.")
