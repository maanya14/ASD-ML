import pandas as pd

crowd = pd.read_csv("crowd_output.csv")
face = pd.read_csv("facial_output.csv")
audio = pd.read_csv("audio_output.csv")
physio = pd.read_csv("physio_output.csv")
visual = pd.read_csv("visual_output.csv")

# -----------------------
# ALIGN LENGTHS (CRITICAL FIX)
# -----------------------
min_len = min(
    len(crowd),
    len(face),
    len(audio),
    len(physio),
    len(visual)
)

crowd = crowd.iloc[:min_len]
face = face.iloc[:min_len]
audio = audio.iloc[:min_len]
physio = physio.iloc[:min_len]
visual = visual.iloc[:min_len]

# -----------------------
# MERGE
# -----------------------
df = pd.DataFrame({
    "crowd_density_norm": crowd["crowd_density_norm"].values,
    "facial_discomfort_prob": face["facial_discomfort_prob"].values,
    "audio_overload_score": audio["audio_overload_score"].values,
    "physio_stress_prob": physio["physio_stress_prob"].values,
    "visual_confusion_prob": visual["overload_prob"].values
})

df.to_csv("everything_aligned.csv", index=False)

print(f"✅ Merged successfully! Rows: {len(df)}")