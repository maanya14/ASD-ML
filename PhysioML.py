import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
# DEFINE MODELS
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# ==============================
# TRAIN & EVALUATE MODELS
# ==============================
results = {}
probabilities = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    results[name] = auc
    probabilities[name] = probs
    roc_data[name] = roc_curve(y_val, probs)

# ==============================
# PRINT MODEL COMPARISON
# ==============================
print("\nMODEL PERFORMANCE COMPARISON")
print("----------------------------------")

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, auc in sorted_results:
    print(f"{name:<20} : ROC-AUC = {auc:.4f}")

# Best Model
best_model_name = sorted_results[0][0]
best_probs = probabilities[best_model_name]

print(f"\nBest Performing Model: {best_model_name}")

# ==============================
# ROC CURVE VISUALIZATION
# ==============================
plt.figure(figsize=(8, 6))

for name in results:
    fpr, tpr, _ = roc_data[name]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Models)")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# AUTO-LEARN THRESHOLDS (BEST MODEL)
# ==============================
thresholds = {
    "low": np.percentile(best_probs, 33),
    "moderate": np.percentile(best_probs, 66),
    "severe": np.percentile(best_probs, 85)
}

print(f"\n{best_model_name} Learned Thresholds:")
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
val_df["overload_probability"] = best_probs
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

compact_df.to_csv("WESAD_OVERLOAD_RESULTS.csv", index=False)

print("\nCompact timeline file saved.")
print(compact_df.head())
print("Shape:", compact_df.shape)