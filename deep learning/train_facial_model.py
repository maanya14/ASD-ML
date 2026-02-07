import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\annotations_all.csv"
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# CREATE TARGET LABEL
# -----------------------------
# 1 = discomfort, 0 = non-discomfort
DISCOMFORT_CLASSES = [3, 4, 5]

df["discomfort"] = df["class_id"].apply(
    lambda x: 1 if x in DISCOMFORT_CLASSES else 0
)

# -----------------------------
# SELECT FEATURES
# -----------------------------
X = df[["x_center", "y_center", "width", "height"]].values
y = df["discomfort"].values

# -----------------------------
# NORMALIZE
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TORCH DATA
# -----------------------------
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

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
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# EVALUATION
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).squeeze()
        predicted = (preds > 0.5).int()
        correct += (predicted == yb.int()).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.3f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "facial_discomfort_model.pth")
print("Facial model trained and saved.")

