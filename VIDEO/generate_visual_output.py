import os
import av
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

VIDEO_FOLDER = "../video_samples"
WINDOW = 10   # must match training

# Load models
light_model = joblib.load("../models/light_model.pkl")
fusion_model = load_model("../models/fusion_model.h5")


def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_video(video_path):
    container = av.open(video_path)
    stream = next((s for s in container.streams if s.type == "video"), None)

    fps = float(stream.average_rate) if stream.average_rate else 24.0

    brightness, contrast, motion, flicker = [], [], [], []
    prev_gray = None
    prev_brightness = None

    # -------- Extract features --------
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b = np.mean(gray)
        c = np.std(gray)

        m = 0 if prev_gray is None else np.mean(cv2.absdiff(gray, prev_gray))
        f = 0 if prev_brightness is None else abs(b - prev_brightness)

        brightness.append(b)
        contrast.append(c)
        motion.append(m)
        flicker.append(f)

        prev_gray = gray
        prev_brightness = b

    # -------- Predict light probability --------
    features = np.column_stack([brightness, contrast, motion, flicker])
    light_probs = light_model.predict(features)

    # -------- Sequence + overload prediction --------
    rows = []
    sequence = []

    for i in range(len(light_probs)):
        sequence.append([light_probs[i]])

        if len(sequence) == WINDOW:
            seq_input = np.array(sequence).reshape(1, WINDOW, 1)
            overload_prob = fusion_model.predict(seq_input, verbose=0)[0][0]

            timestamp = to_timestamp(i / fps)

            rows.append({
                "timestamp": timestamp,
                "light_prob": float(light_probs[i]),
                "overload_prob": float(overload_prob)
            })

            sequence.pop(0)

    return rows


# -------- Process all videos --------
all_rows = []

for file in os.listdir(VIDEO_FOLDER):
    if file.endswith((".mp4", ".MOV", ".mov")):
        print("Processing:", file)
        path = os.path.join(VIDEO_FOLDER, file)

        rows = process_video(path)
        for r in rows:
            r["video"] = file

        all_rows.extend(rows)

# Save output
os.makedirs("data", exist_ok=True)

df = pd.DataFrame(all_rows)
df.to_csv("data/visual_output.csv", index=False)

print("✅ visual_output.csv generated")