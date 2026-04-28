# This script: Trains a neural network to predict a soft visual overload probability based on crowd density and facial discomfort.

# It produces:
# - overload_prob_model.pth
# - Which your earlier inference script uses.


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fusion_model import AttentionFusion

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("everything_aligned.csv")

feat_cols = [
    "crowd_density_norm",
    "facial_discomfort_prob",
    "audio_overload_score",
    "physio_stress_prob"
]

X = df[feat_cols].values

# Pseudo target (until real labels available)
Y = (
    df["crowd_density_norm"] +
    df["facial_discomfort_prob"] +
    df["audio_overload_score"] +
    df["physio_stress_prob"]
) / 4

Y = (1 / (1 + np.exp(-Y))).values.reshape(-1, 1)

# -----------------------
# TORCH DATA
# -----------------------
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------
# MODEL
# -----------------------
model = AttentionFusion()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# TRAIN
# -----------------------
EPOCHS = 50

for epoch in range(EPOCHS):
    total_loss = 0

    for xb, yb in loader:
        optimizer.zero_grad()

        outputs, _ = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# -----------------------
# SAVE
# -----------------------
torch.save(model.state_dict(), "overload_prob_model.pth")
print("Model saved!")