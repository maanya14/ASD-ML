import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PATHS (adjust if needed)
# -----------------------------
CROWD_CSV = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\csv_out\all_sets_combined.csv"
FACIAL_CSV = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\annotations_all.csv"

OUTPUT_CSV = r"C:\Users\shrey\OneDrive\Desktop\ASD-ML\everything_aligned.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
crowd_df = pd.read_csv(CROWD_CSV)
face_df = pd.read_csv(FACIAL_CSV)

# -----------------------------
# CROWD FEATURE
# -----------------------------
# Use frame as time index
crowd_feat = crowd_df.groupby("frame")["people_count"].mean().reset_index()
crowd_feat.rename(columns={"people_count": "crowd_density"}, inplace=True)

# -----------------------------
# FACIAL DISCOMFORT FEATURE
# -----------------------------
DISCOMFORT_CLASSES = [3, 4, 5]

face_df["discomfort"] = face_df["class_id"].apply(
    lambda x: 1 if x in DISCOMFORT_CLASSES else 0
)

# Aggregate per image
face_feat = face_df.groupby("image")["discomfort"].mean().reset_index()

# Extract frame number safely
face_feat["frame"] = face_feat["image"].str.extract(r"(\d+)")

# DROP rows where frame is missing
face_feat = face_feat.dropna(subset=["frame"])

# Convert to int AFTER dropping NaNs
face_feat["frame"] = face_feat["frame"].astype(int)

face_feat.rename(columns={"discomfort": "facial_discomfort"}, inplace=True)


# -----------------------------
# MERGE FEATURES
# -----------------------------
df = pd.merge(crowd_feat, face_feat[["frame", "facial_discomfort"]],
              on="frame", how="left")

df.fillna(0, inplace=True)

# -----------------------------
# NORMALIZE
# -----------------------------
scaler = MinMaxScaler()

df[["crowd_density_norm", "facial_discomfort_norm"]] = scaler.fit_transform(
    df[["crowd_density", "facial_discomfort"]]
)

# -----------------------------
# SAVE
# -----------------------------
df_out = df[["frame", "crowd_density_norm", "facial_discomfort_norm"]]
df_out.to_csv(OUTPUT_CSV, index=False)

print("everything_aligned.csv created successfully.")
print(df_out.head())
