import pandas as pd

crowd = pd.read_csv("crowd_output.csv")
face = pd.read_csv("facial_output.csv")
audio = pd.read_csv("audio_output.csv")
physio = pd.read_csv("physio_output.csv")

df = pd.DataFrame()

df["crowd_density_norm"] = crowd["crowd_density_norm"]
df["facial_discomfort_prob"] = face["facial_discomfort_prob"]
df["audio_overload_score"] = audio["audio_overload_score"]
df["physio_stress_prob"] = physio["physio_stress_prob"]

df.to_csv("everything_aligned.csv", index=False)

print("Merged successfully!")