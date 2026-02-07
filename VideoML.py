import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\csv_out\all_sets_combined.csv"
SEQUENCE_LENGTH = 30   # ~1 second if 30 FPS
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)

# Sort by dataset and frame
df = df.sort_values(["dataset", "frame"])

# Use people_count as main feature
values = df["people_count"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# -------------------------
# CREATE SEQUENCES
# -------------------------
def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

X = create_sequences(values_scaled, SEQUENCE_LENGTH)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# MODEL
# -------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        reconstructed, _ = self.decoder(hidden)
        return reconstructed

model = LSTMAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# TRAINING
# -------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, _ in loader:
        optimizer.zero_grad()
        recon = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.6f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "crowd_overload_lstm.pth")
print("Model training complete and saved.")
