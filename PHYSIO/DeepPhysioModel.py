import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

# ==============================
# PARAMETERS
# ==============================
FS_TARGET = 4
WINDOW_SEC = 30
STEP_SEC = 10

WINDOW = FS_TARGET * WINDOW_SEC
STEP = FS_TARGET * STEP_SEC

DATASET_DIR = "../Dataset"

# ==============================
# RESAMPLING
# ==============================
def resample_signal(signal, orig_fs, target_fs):
    factor = int(orig_fs / target_fs)
    return signal[::factor]

# ==============================
# RAW DATA PROCESSING
# ==============================
def process_subject_raw(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        subject_id = data["subject"]
        wrist = data["signal"]["wrist"]

        eda = wrist["EDA"]
        temp = wrist["TEMP"]
        acc = wrist["ACC"]
        bvp = wrist["BVP"]
        label = data["label"]

        eda_ds = eda
        temp_ds = temp
        acc_ds = resample_signal(acc, 32, 4)
        bvp_ds = resample_signal(bvp, 64, 4)
        label_ds = resample_signal(label, 700, 4)

        min_len = min(len(eda_ds), len(temp_ds), len(acc_ds), len(bvp_ds), len(label_ds))

        eda_ds = eda_ds[:min_len]
        temp_ds = temp_ds[:min_len]
        acc_ds = acc_ds[:min_len]
        bvp_ds = bvp_ds[:min_len]
        label_ds = label_ds[:min_len]

        valid_idx = np.where((label_ds == 1) | (label_ds == 2))[0]

        eda_ds = eda_ds[valid_idx]
        temp_ds = temp_ds[valid_idx]
        acc_ds = acc_ds[valid_idx]
        bvp_ds = bvp_ds[valid_idx]
        label_ds = label_ds[valid_idx]

        label_ds = np.where(label_ds == 2, 1, 0)

        sequences, labels, groups = [], [], []

        for start in range(0, len(eda_ds) - WINDOW, STEP):
            end = start + WINDOW

            window = np.column_stack([
                eda_ds[start:end],
                temp_ds[start:end],
                bvp_ds[start:end],
                np.linalg.norm(acc_ds[start:end], axis=1)
            ])

            window_label = int(np.mean(label_ds[start:end]) > 0.3)

            sequences.append(window)
            labels.append(window_label)
            groups.append(subject_id)

        return sequences, labels, groups

    except Exception as e:
        print("Error:", e)
        return [], [], []

# ==============================
# LOAD DATA
# ==============================
all_X, all_y, all_groups = [], [], []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".pkl"):
        X_seq, y_seq, groups = process_subject_raw(os.path.join(DATASET_DIR, file))
        all_X.extend(X_seq)
        all_y.extend(y_seq)
        all_groups.extend(groups)

X = np.array(all_X)
y = np.array(all_y)
groups = np.array(all_groups)

print("Data Shape:", X.shape)

# ==============================
# FILTER BAD SUBJECTS
# ==============================
subject_labels = defaultdict(list)

for label, group in zip(y, groups):
    subject_labels[group].append(label)

valid_subjects = []

for subject in subject_labels:
    if len(np.unique(subject_labels[subject])) > 1:
        valid_subjects.append(subject)

mask = np.isin(groups, valid_subjects)

X = X[mask]
y = y[mask]
groups = groups[mask]

# ==============================
# SPLIT
# ==============================
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in sgkf.split(X, y, groups):
    break

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("Train:", np.bincount(y_train))
print("Val:", np.bincount(y_val))

# ==============================
# NORMALIZATION
# ==============================
mean = X_train.mean()
std = X_train.std() + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# ==============================
# MODEL
# ==============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class Model(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.conv1 = nn.Conv1d(n_features, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 5)

        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        self.attn = Attention(128)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        context, weights = self.attn(x)

        return self.fc(context), weights

# ==============================
# TRAINING SETUP
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(X.shape[2]).to(device)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

pos_weight = torch.tensor([
    (len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-6)
], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAIN LOOP
# ==============================
loss_history = []

for epoch in range(15):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        logits, _ = model(xb)
        logits = logits.squeeze()

        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==============================
# LOSS CURVE
# ==============================
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# ==============================
# EVALUATION
# ==============================
model.eval()
with torch.no_grad():
    logits, attn_weights = model(X_val_t)
    logits = logits.squeeze()
    preds = torch.sigmoid(logits).cpu().numpy()
    attn_weights = attn_weights.cpu().numpy()

auc = roc_auc_score(y_val, preds)
print("\nROC-AUC:", auc)

# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(y_val, preds)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()

# ==============================
# ATTENTION VISUALIZATION
# ==============================
sample_idx = 0

attention = attn_weights[sample_idx].squeeze()
signal = X_val[sample_idx]

plt.figure(figsize=(10,4))
plt.plot(attention)
plt.title("Attention Weights Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Importance")
plt.grid()
plt.show()

# ==============================
# OVERLAY ATTENTION ON EDA
# ==============================
eda_signal = signal[:, 0]

plt.figure(figsize=(12,4))
plt.plot(eda_signal, label="EDA")
plt.plot(attention * np.max(eda_signal), label="Attention")
plt.legend()
plt.title("EDA + Attention Overlay")
plt.grid()
plt.show()


# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "model.pth")

# Save normalization values
np.save("mean.npy", mean)
np.save("std.npy", std)

print("\nModel and normalization saved!")