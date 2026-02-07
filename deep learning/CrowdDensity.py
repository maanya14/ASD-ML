import os
import pandas as pd
from scipy.io import loadmat

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "."          # folder containing .vbb files
OUTPUT_DIR = "csv_out"  # output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data = []  # to store combined results

# ----------------------------
# PROCESS EACH VBB FILE
# ----------------------------
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".vbb"):
        print(f"Processing {file} ...")

        file_path = os.path.join(DATA_DIR, file)
        mat = loadmat(file_path)

        # Caltech VBB structure
        vbb = mat['A'][0][0]
        obj_lists = vbb[1][0]   # object lists per frame

        rows = []

        for frame_id, objs in enumerate(obj_lists):
            if objs.size == 0:
                people_count = 0
            else:
                people_count = len(objs[0])

            rows.append({
                "dataset": file,
                "frame": frame_id,
                "people_count": people_count
            })

        df = pd.DataFrame(rows)

        # Save individual CSV
        csv_name = file.replace(".vbb", ".csv")
        df.to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)

        all_data.append(df)

# ----------------------------
# SAVE COMBINED CSV
# ----------------------------
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(os.path.join(OUTPUT_DIR, "all_sets_combined.csv"), index=False)

print("\n Conversion complete!")
print(f" CSV files saved in: {OUTPUT_DIR}")
