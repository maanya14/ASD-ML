import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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
# TRAIN MODEL
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# VALIDATION PROBABILITIES
# ==============================
val_probs = model.predict_proba(X_val)[:, 1]

print("Validation ROC-AUC:",
      roc_auc_score(y_val, val_probs))

# ==============================
# AUTO-LEARN THRESHOLDS
# ==============================
thresholds = {
    "low": np.percentile(val_probs, 33),
    "moderate": np.percentile(val_probs, 66),
    "severe": np.percentile(val_probs, 85)
}

print("\nLearned thresholds:")
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
val_df["overload_probability"] = val_probs
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


#GRAPH PLOTS

# # LOAD COMPACT DATA
# # ==============================
# df = pd.read_csv("WESAD_OVERLOAD_RESULTS.csv")

# # ==============================
# # COLOR MAP FOR OVERLOAD LEVELS
# # ==============================
# level_colors = {
#     "Low": "green",
#     "Moderate": "orange",
#     "High": "red",
#     "Severe": "darkred"
# }

# # ==============================
# # PLOT PER SUBJECT
# # ==============================
# subjects = df["subject_id"].unique()

# for subject in subjects:
#     sub_df = df[df["subject_id"] == subject]

#     plt.figure(figsize=(12, 4))

#     # Plot overload probability
#     for level, color in level_colors.items():
#         mask = sub_df["overload_level"] == level
#         plt.scatter(
#             sub_df.loc[mask, "center_time_sec"],
#             sub_df.loc[mask, "overload_probability"],
#             label=level,
#             color=color,
#             s=30
#         )

#     # Optional: Ground truth shading
#     stress_mask = sub_df["label"] == 1
#     plt.fill_between(
#         sub_df["center_time_sec"],
#         0, 1,
#         where=stress_mask,
#         color="gray",
#         alpha=0.15,
#         label="Ground Truth Stress"
#     )

#     plt.title(f"Subject {subject} – Sensory Overload Timeline")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Overload Probability")
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()