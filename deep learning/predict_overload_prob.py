import torch
import pandas as pd
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
class OverloadNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = OverloadNet()
model.load_state_dict(torch.load("overload_prob_model.pth"))
model.eval()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("everything_aligned.csv")

features = df[["crowd_density_norm", "facial_discomfort_norm"]].values
X = torch.tensor(features, dtype=torch.float32)

# -----------------------------
# PREDICT PROBABILITY
# -----------------------------
with torch.no_grad():
    probs = model(X).squeeze().numpy()

df["overload_probability"] = probs

# -----------------------------
# SAVE RESULTS
# -----------------------------
df.to_csv("overload_probabilities.csv", index=False)

print("Overload probabilities computed and saved.")
print(df.head())

def level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.6:
        return "Moderate"
    else:
        return "High"

df["overload_level"] = df["overload_probability"].apply(level)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["frame"], df["overload_probability"])
plt.xlabel("Frame")
plt.ylabel("Probability of Overload")
plt.title("Sensory Overload Probability Over Time")
plt.show()
