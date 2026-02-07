import os
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')


ROOT_DIR = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\YOLO_format"
OUTPUT_CSV = "annotations_all.csv"

rows = []

# auto-detect dataset splits
for split in os.listdir(ROOT_DIR):
    split_path = os.path.join(ROOT_DIR, split)
    if not os.path.isdir(split_path):
        continue

    labels_dir = os.path.join(split_path, "labels")
    images_dir = os.path.join(split_path, "images")

    if not os.path.exists(labels_dir):
        continue

    print(f"Reading labels from: {labels_dir}")

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue  # empty annotation file

        image_name = label_file.replace(".txt", ".jpg")  # change if png

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = parts

            rows.append({
                "split": split,
                "image": image_name,
                "class_id": int(class_id),
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width": float(width),
                "height": float(height)
            })

# Save CSV
df = pd.DataFrame(rows)

if df.empty:
    print("⚠️ No annotations found. CSV will be empty.")
else:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Conversion complete. Saved {len(df)} rows to {OUTPUT_CSV}")
