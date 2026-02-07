import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# CONFIG
# -------------------------------
DATA_CSV = "everything_aligned.csv"  # merged dataset (crowd + face + brightness + flicker)
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

# -------------------------------
# LOAD & PREPROCESS
# -------------------------------
df = pd.read_csv(DATA_CSV)

# Features
feat_cols = [
    "crowd_density_norm",
    "facial_discomfort_norm"
]


X = df[feat_cols].values

# Create soft target overload score using rule
# Adjust weights as you see fit
alpha, beta = 0.6, 0.4

combined = (
    alpha * df["crowd_density_norm"] +
    beta * df["facial_discomfort_norm"]
)

overload_score = 1 / (1 + np.exp(-combined))

# Target
Y = overload_score.astype(np.float32).values.reshape(-1, 1)

# Convert numpy → PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# MODEL
# -------------------------------
class OverloadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(feat_cols), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        return self.net(x)

model = OverloadNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# TRAINING LOOP
# -------------------------------
print("Training overload probability model...")

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

print("Training complete.")

# -------------------------------
# SAVE MODEL
# -------------------------------
torch.save(model.state_dict(), "overload_prob_model.pth")
print("Model saved.")
