import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        x_weighted = x * weights
        out = self.fc(x_weighted)
        return out, weights