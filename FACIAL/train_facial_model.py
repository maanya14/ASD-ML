# This script:
# - Trains a neural network to classify whether a detected face represents discomfort or not.

# It produces:
# - facial_discomfort_model.pth

# Which later feeds into your visual overload probability model.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "annotations_all.csv"
MODEL_PATH = "facial_discomfort_model.pth"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

DISCOMFORT_CLASSES = [3, 4, 5]

df["discomfort"] = df["class_id"].apply(
    lambda x: 1 if x in DISCOMFORT_CLASSES else 0
)

X = df[["x_center", "y_center", "width", "height"]].values
y = df["discomfort"].values

# -----------------------------
# NORMALIZE
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TORCH
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# -----------------------------
# MODEL
# -----------------------------
class FacialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = FacialMLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAIN
# -----------------------------
for epoch in range(20):
    optimizer.zero_grad()

    preds = model(X_train).squeeze()
    loss = criterion(preds, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)

# -----------------------------
# GENERATE PROBABILITIES
# -----------------------------
model.eval()

with torch.no_grad():
    probs = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

df["facial_discomfort_prob"] = probs

df[["facial_discomfort_prob"]].to_csv("facial_output.csv", index=False)

print("Facial output saved!")