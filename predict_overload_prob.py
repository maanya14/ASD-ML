# This script:
# - Loads a trained neural network
# - Feeds crowd + facial features
# - Computes overload probability per frame
# - Saves and visualizes results

# It generates:
# - overload_probabilities.csv

# Which contains:
# - frame
# - crowd_density_norm
# - facial_discomfort_norm
# - overload_probability
# - overload_level

import torch
import pandas as pd
import numpy as np

from fusion_model import AttentionFusion
from alert_system import trigger_alert
from explainability import plot_contributions, plot_overload

# -----------------------
# LOAD MODEL
# -----------------------
model = AttentionFusion()
model.load_state_dict(torch.load("overload_prob_model.pth"))
model.eval()

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("everything_aligned.csv")

features = df[
    [
        "crowd_density_norm",
        "facial_discomfort_prob",
        "audio_overload_score",
        "physio_stress_prob"
    ]
].values

X = torch.tensor(features, dtype=torch.float32)

# -----------------------
# PREDICT
# -----------------------
with torch.no_grad():
    outputs, weights = model(X)

probs = outputs.squeeze().numpy()
weights = weights.numpy()

df["overload_probability"] = probs

# -----------------------
# LEVELS
# -----------------------
def level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.6:
        return "Moderate"
    else:
        return "High"

df["overload_level"] = df["overload_probability"].apply(level)

# -----------------------
# EXPLAINABILITY
# -----------------------
df["crowd_weight"] = weights[:, 0]
df["face_weight"] = weights[:, 1]
df["audio_weight"] = weights[:, 2]
df["physio_weight"] = weights[:, 3]

# -----------------------
# ALERT SYSTEM
# -----------------------
for p in df["overload_probability"]:
    trigger_alert(p)

# -----------------------
# SAVE
# -----------------------
df.to_csv("multimodal_output.csv", index=False)

# -----------------------
# PLOTS
# -----------------------
plot_contributions(df)
plot_overload(df)

print("Prediction complete!")