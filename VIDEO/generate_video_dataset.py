import pandas as pd
import joblib

# Load features
df = pd.read_csv("../data/features.csv")

# Load trained light model
light_model = joblib.load("../models/light_model.pkl")

# Input features for light model
X = df[["brightness", "contrast", "motion", "flicker"]]

# 🔥 THIS is what you asked about
df["light_prob"] = light_model.predict(X)

# ---- Create overload label (temporary for training) ----
df["overload"] = (
    (df["light_prob"] > 0.6) | 
    (df["motion"] > df["motion"].mean())
).astype(int)

# Save
df.to_csv("../data/fusion_dataset.csv", index=False)

print("fusion_dataset.csv created")