import streamlit as st
import numpy as np
import pickle
import time
import torch
import os

# ==============================
# LOAD MODEL
# ==============================
from inference import predict_overload

# ==============================
# LOAD PICKLE DATA (FIXED)
# ==============================
@st.cache_data
def load_data(selected_file):
    import os
    import numpy as np
    import pickle

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "Dataset")
    file_path = os.path.join(dataset_path, selected_file)

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]

    eda = wrist["EDA"]      # 4 Hz
    temp = wrist["TEMP"]    # 4 Hz
    acc = wrist["ACC"]      # 32 Hz
    bvp = wrist["BVP"]      # 64 Hz

    # ==============================
    # 🔥 RESAMPLING (KEY FIX)
    # ==============================
    acc_ds = acc[::8]   # 32 → 4 Hz
    bvp_ds = bvp[::16]  # 64 → 4 Hz

    # ==============================
    # ALIGN LENGTHS
    # ==============================
    min_len = min(len(eda), len(temp), len(acc_ds), len(bvp_ds))

    eda = eda[:min_len]
    temp = temp[:min_len]
    acc_ds = acc_ds[:min_len]
    bvp_ds = bvp_ds[:min_len]

    # Magnitude of acceleration
    acc_mag = np.linalg.norm(acc_ds, axis=1)

    # ==============================
    # FINAL STACK (NOW SAFE)
    # ==============================
    signal = np.column_stack([
        eda,
        temp,
        bvp_ds,
        acc_mag
    ])

    return signal

# ==============================
# UI SETUP
# ==============================
st.set_page_config(page_title="Sensory Overload Monitor", layout="wide")

st.title("🧠 Real-Time Sensory Overload Detection")

# ==============================
# FILE SELECTOR (NEW 🔥)
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "Dataset")

if not os.path.exists(dataset_path):
    st.error("❌ Dataset folder not found!")
    st.stop()

files = [f for f in os.listdir(dataset_path) if f.endswith(".pkl")]

if not files:
    st.error("❌ No .pkl files found in Dataset folder!")
    st.stop()

selected_file = st.selectbox("Select Subject File", files)

signal = load_data(selected_file)

# ==============================
# UI COMPONENTS
# ==============================
col1, col2 = st.columns(2)

chart = col1.line_chart()
status_box = col2.empty()
prob_box = col2.empty()

# ==============================
# REAL-TIME SIMULATION
# ==============================
buffer = []

for i in range(len(signal)):

    buffer.append(signal[i])

    if len(buffer) > 120:
        buffer.pop(0)

    # Graph (EDA only)
    eda_vals = [x[0] for x in buffer]
    chart.add_rows(np.array(eda_vals[-1:]).reshape(1, -1))

    if len(buffer) == 120:
        window = np.array(buffer)

        prob, level = predict_overload(window)

        prob_box.metric("Overload Probability", f"{prob:.3f}")

        if level == "Low":
            status_box.success(f"🟢 {level}")
        elif level == "Moderate":
            status_box.warning(f"🟡 {level}")
        elif level == "High":
            status_box.warning(f"🟠 {level}")
        else:
            status_box.error(f"🔴 {level}")

    time.sleep(0.25)