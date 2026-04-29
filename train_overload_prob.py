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
    "physio_stress_prob",
    "visual_confusion_prob"
]

X = df[feat_cols].values

# -----------------------
# TARGET
# -----------------------
Y = np.mean(X, axis=1)
Y = (1 / (1 + np.exp(-Y))).reshape(-1, 1)

# -----------------------
# TORCH
# -----------------------
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------
# MODEL (FIXED)
# -----------------------
input_dim = X.shape[1]   # 🔥 dynamic
model = AttentionFusion(input_dim=input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# TRAIN
# -----------------------
EPOCHS = 30

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
print("✅ Model saved!")