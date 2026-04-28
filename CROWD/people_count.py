# What It Does: 
# - Loads all_sets_combined.csv
# - Uses only people_count
# - Creates 30-frame sequences
# - Trains an LSTM Autoencoder
# - Learns to reconstruct crowd patterns
# - Saves model as crowd_overload_lstm.pth

# What Type of Model?
# - Unsupervised anomaly detection.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "csv_out/all_sets_combined.csv"
MODEL_PATH = "crowd_overload_lstm.pth"
SEQ_LEN = 30

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)
df = df.sort_values(["dataset", "frame"])

values = df["people_count"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# -------------------------
# CREATE SEQUENCES
# -------------------------
def create_sequences(data, seq_len):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

X = create_sequences(values_scaled, SEQ_LEN)
X_tensor = torch.tensor(X, dtype=torch.float32)

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
        out, _ = self.decoder(hidden)
        return out

model = LSTMAutoencoder()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# -------------------------
# INFERENCE (ANOMALY SCORE)
# -------------------------
errors = []

with torch.no_grad():
    for seq in X_tensor:
        seq = seq.unsqueeze(0)
        recon = model(seq)

        error = torch.mean((recon - seq) ** 2).item()
        errors.append(error)

# Pad initial frames
errors = [errors[0]] * SEQ_LEN + errors

df["crowd_density_norm"] = errors[:len(df)]

# Normalize final output
df["crowd_density_norm"] = (
    df["crowd_density_norm"] - df["crowd_density_norm"].min()
) / (df["crowd_density_norm"].max() - df["crowd_density_norm"].min())

# -------------------------
# SAVE OUTPUT
# -------------------------
df[["frame", "crowd_density_norm"]].to_csv("crowd_output.csv", index=False)

print("Crowd output saved!")