import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==============================
# PARAMETERS
# ==============================
FS = 4                 # Sampling frequency after alignment (Hz)
WINDOW_SEC = 30        # Window length in seconds
STEP_SEC = 10          # Step size in seconds

WINDOW = WINDOW_SEC * FS
STEP = STEP_SEC * FS

DATASET_DIR = "./Dataset"

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(eda, temp, acc, bvp):
    features = {}

    
    eda = np.asarray(eda).squeeze()
    temp = np.asarray(temp).squeeze()
    bvp = np.asarray(bvp).squeeze()

    # EDA features
    features["eda_mean"] = np.mean(eda)
    features["eda_std"] = np.std(eda)
    features["eda_slope"] = np.polyfit(range(len(eda)), eda, 1)[0]
    features["eda_peaks"] = len(find_peaks(eda, height=np.mean(eda))[0])

    # Temperature features
    features["temp_mean"] = np.mean(temp)
    features["temp_std"] = np.std(temp)
    features["temp_slope"] = np.polyfit(range(len(temp)), temp, 1)[0]

    # ACC magnitude features
    acc_mag = np.linalg.norm(acc, axis=1)
    features["acc_mean"] = np.mean(acc_mag)
    features["acc_std"] = np.std(acc_mag)
    features["acc_max"] = np.max(acc_mag)

    # BVP features
    features["bvp_std"] = np.std(bvp)

    return features

# ==============================
# PROCESS ONE SUBJECT
# ==============================
def process_subject(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        subject_id = data["subject"]
        wrist = data["signal"]["wrist"]

        eda = wrist["EDA"]
        temp = wrist["TEMP"]
        acc = wrist["ACC"]
        bvp = wrist["BVP"]
        label = data["label"]

        # ==============================
        # DOWNSAMPLING
        # ==============================
        acc_ds = acc[::8]        # 32 Hz → 4 Hz
        bvp_ds = bvp[::16]       # 64 Hz → 4 Hz
        label_ds = label[::175]  # Label alignment

        # ==============================
        # KEEP BASELINE (1) & STRESS (2)
        # ==============================
        valid_idx = np.where((label_ds == 1) | (label_ds == 2))[0]

        if len(valid_idx) < WINDOW:
            print(f"Skipping {subject_id}: not enough valid data")
            return None

        eda = eda[valid_idx]
        temp = temp[valid_idx]
        acc = acc_ds[valid_idx]
        bvp = bvp_ds[valid_idx]
        label_ds = np.where(label_ds[valid_idx] == 2, 1, 0)

        rows = []

        # ==============================
        # SLIDING WINDOW FEATURE EXTRACTION
        # ==============================
        for start in range(0, len(eda) - WINDOW, STEP):
            end = start + WINDOW

            feats = extract_features(
                eda[start:end],
                temp[start:end],
                acc[start:end],
                bvp[start:end]
            )

            # ==============================
            # TIMESTAMPS (seconds)
            # ==============================
            feats["start_time_sec"] = start / FS
            feats["end_time_sec"] = end / FS
            feats["center_time_sec"] = (start + end) / (2 * FS)

            # Label & subject info
            feats["label"] = int(np.round(np.mean(label_ds[start:end])))
            feats["subject_id"] = subject_id

            rows.append(feats)

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return None

# ==============================
# PROCESS ALL SUBJECTS
# ==============================
all_subject_dfs = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".pkl"):
        print(f"Processing {file}...")
        df = process_subject(os.path.join(DATASET_DIR, file))
        if df is not None and not df.empty:
            all_subject_dfs.append(df)

# ==============================
# COMBINE & SAVE FINAL DATASET
# ==============================
final_df = pd.concat(all_subject_dfs, ignore_index=True)

print("\nFinal dataset shape:", final_df.shape)
print(final_df["label"].value_counts())

final_df.to_csv("WESAD_DATA.csv", index=False)
