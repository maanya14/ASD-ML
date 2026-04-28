import os
import pandas as pd
from extract_visual_features import analyze_video

VIDEO_FOLDER = "../video_samples"
OUTPUT = "../data/features.csv"

all_rows = []

for file in os.listdir(VIDEO_FOLDER):
    if file.endswith((".mp4", ".MOV", ".mov")):
        path = os.path.join(VIDEO_FOLDER, file)
        print("Processing:", file)

        rows = analyze_video(path, file)
        all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT, index=False)

print("Saved features:", OUTPUT)