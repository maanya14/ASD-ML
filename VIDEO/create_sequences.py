import numpy as np
import pandas as pd

df = pd.read_csv("../data/fusion_dataset.csv")

FEATURES = ["light_prob"]   # only light for now
WINDOW = 10

X, y = [], []

for i in range(len(df) - WINDOW):
    seq = df[FEATURES].iloc[i:i+WINDOW].values
    label = df["overload"].iloc[i+WINDOW]

    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

np.save("../data/X.npy", X)
np.save("../data/y.npy", y)

print("Sequences created:", X.shape)