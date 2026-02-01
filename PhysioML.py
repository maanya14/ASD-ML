import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("WESAD_DATA.csv")

# ==============================
# FEATURE / LABEL SPLIT
# ==============================
X = df.drop(columns=[
    "label",
    "subject_id",
    "start_time_sec",
    "end_time_sec",
    "center_time_sec"
])

y = df["label"]
groups = df["subject_id"]

# ==============================
# SUBJECT-WISE TRAIN / VALIDATION SPLIT
# ==============================
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(gss.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ==============================
# TRAIN XGBOOST
# ==============================
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_probs)

# ==============================
# TRAIN RANDOM FOREST
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_val_probs = rf_model.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_val_probs)

# ==============================
# MODEL COMPARISON
# ==============================
print("\nMODEL PERFORMANCE COMPARISON")
print("-----------------------------")
print(f"XGBoost ROC-AUC      : {xgb_auc:.4f}")
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")

# ==============================
# ROC CURVE VISUALIZATION
# ==============================
xgb_fpr, xgb_tpr, _ = roc_curve(y_val, xgb_val_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_val, rf_val_probs)

plt.figure(figsize=(8, 6))
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# AUTO-LEARN THRESHOLDS (XGBOOST)
# ==============================
thresholds = {
    "low": np.percentile(xgb_val_probs, 33),
    "moderate": np.percentile(xgb_val_probs, 66),
    "severe": np.percentile(xgb_val_probs, 85)
}

print("\nXGBoost Learned Thresholds:")
for k, v in thresholds.items():
    print(f"{k}: {v:.3f}")

# ==============================
# ASSIGN OVERLOAD LEVELS
# ==============================
def assign_overload_level(p, th):
    if p < th["low"]:
        return "Low"
    elif p < th["moderate"]:
        return "Moderate"
    elif p < th["severe"]:
        return "High"
    else:
        return "Severe"

# ==============================
# APPLY TO VALIDATION DATA
# ==============================
val_df = df.iloc[val_idx].copy()
val_df["overload_probability"] = xgb_val_probs
val_df["overload_level"] = val_df["overload_probability"].apply(
    lambda p: assign_overload_level(p, thresholds)
)

# ==============================
# SAVE COMPACT TIMELINE DATA
# ==============================
compact_df = val_df[[
    "subject_id",
    "start_time_sec",
    "end_time_sec",
    "center_time_sec",
    "overload_probability",
    "overload_level",
    "label"
]]

compact_df.to_csv(
    "WESAD_OVERLOAD_RESULTS.csv",
    index=False
)

print("\nCompact timeline file saved.")
print(compact_df.head())
print("Shape:", compact_df.shape)
