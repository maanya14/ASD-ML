import torch
import torch.nn as nn
import numpy as np

# ==============================
# LOAD NORMALIZATION
# ==============================
mean = np.load("mean.npy")
std = np.load("std.npy")

# ==============================
# MODEL (same as training)
# ==============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

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

        x = self.attn(x)

        return torch.sigmoid(self.fc(x))

# ==============================
# LOAD MODEL
# ==============================
model = Model(n_features=4)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_overload(signal_window):
    """
    signal_window shape: (time_steps, 4)
    """

    # Normalize
    signal_window = (signal_window - mean) / std

    # Convert to tensor
    x = torch.tensor(signal_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    # Convert to level
    if prob < 0.25:
        level = "Low"
    elif prob < 0.5:
        level = "Moderate"
    elif prob < 0.75:
        level = "High"
    else:
        level = "Severe"

    return prob, level


if __name__ == "__main__":
    # Fake example input
    sample = np.random.randn(120, 4)

    prob, level = predict_overload(sample)

    print("Overload Probability:", prob)
    print("Overload Level:", level)