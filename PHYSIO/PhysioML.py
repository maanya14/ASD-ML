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
# SUBJECT-WISE SPLIT
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
# SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ALSO SCALE FULL DATA (IMPORTANT FOR FINAL OUTPUT)
X_all = scaler.transform(X)

# ==============================
# MODELS
# ==============================
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "GradientBoost": GradientBoostingClassifier(),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    )
}

results = {}
roc_data = {}

# ==============================
# TRAIN + EVALUATE
# ==============================
best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)

    results[name] = auc
    roc_data[name] = roc_curve(y_val, probs)

    print(f"{name} AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model

print("\n✅ BEST MODEL SELECTED:", best_model)
print(f"Best AUC: {best_auc:.4f}")

# ==============================
# GENERATE FINAL PROBABILITIES (FULL DATA)
# ==============================
physio_probs = best_model.predict_proba(X_all)[:, 1]

# Normalize (important for fusion)
physio_probs = (physio_probs - physio_probs.min()) / (
    physio_probs.max() - physio_probs.min()
)

# ==============================
# SAVE OUTPUT (IMPORTANT)
# ==============================
output_df = pd.DataFrame({
    "physio_stress_prob": physio_probs
})

output_df.to_csv("physio_output.csv", index=False)

print("✅ physio_output.csv saved!")

# ==============================
# OPTIONAL: ROC PLOT
# ==============================
plt.figure(figsize=(8, 6))

for name in results:
    fpr, tpr, _ = roc_data[name]
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()

plt.show()

# ==============================
# OPTIONAL: SAVE VALIDATION OUTPUT
# ==============================
val_df = df.iloc[val_idx].copy()

val_probs = best_model.predict_proba(X_val)[:, 1]

val_df["physio_stress_prob"] = val_probs

val_df.to_csv("physio_validation_output.csv", index=False)

print("Validation output saved!")