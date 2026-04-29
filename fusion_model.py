import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, input_dim=5):   # 🔥 updated
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 16),   # 5 → 16
            nn.ReLU(),
            nn.Linear(16, input_dim),   # 16 → 5
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),   # 5 → 32
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        x_weighted = x * weights
        out = self.fc(x_weighted)
        return out, weights