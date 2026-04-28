import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv("../data/features.csv")

X = df[["brightness", "contrast", "motion", "flicker"]]

# pseudo target (ONLY for initial learning)
y = (df["flicker"] + df["contrast"] + df["motion"]) / 3

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPRegressor(hidden_layer_sizes=(32,16))
model.fit(X_train, y_train)

joblib.dump(model, "../models/light_model.pkl")

print("Light model trained")