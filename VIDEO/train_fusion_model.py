import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

model = Sequential([
    LSTM(32, input_shape=(X.shape[1], X.shape[2])),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=10, batch_size=16)

model.save("../models/fusion_model.h5")

print("Fusion model trained")