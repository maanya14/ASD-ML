import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("everything_aligned.csv")
df = df.sort_values("frame").reset_index(drop=True)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
WINDOW = 5

df["crowd_mean"] = df["crowd_density_norm"].rolling(WINDOW).mean()
df["crowd_diff"] = df["crowd_density_norm"].diff()
df["face_mean"] = df["facial_discomfort_norm"].rolling(WINDOW).mean()

df.fillna(0, inplace=True)

X = df[
    [
        "crowd_density_norm",
        "crowd_mean",
        "crowd_diff",
        "facial_discomfort_norm",
        "face_mean"
    ]
]

# -----------------------------
# CONTINUOUS OVERLOAD TARGET
# -----------------------------
raw_score = (
    0.6 * df["crowd_density_norm"] +
    0.4 * df["facial_discomfort_norm"]
)

y = 1 / (1 + np.exp(-raw_score))  # probability target

# -----------------------------
# SCALE
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# -----------------------------
# TRAIN REGRESSION MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = np.clip(model.predict(X_test), 0, 1)
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------
# SAVE
# -----------------------------
import joblib
joblib.dump(model, "overload_regression_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("Classical regression model trained successfully.")
