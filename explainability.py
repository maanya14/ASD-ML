import matplotlib.pyplot as plt

def plot_contributions(df):
    plt.figure(figsize=(10,6))

    plt.plot(df["crowd_weight"], label="Crowd")
    plt.plot(df["face_weight"], label="Face")
    plt.plot(df["audio_weight"], label="Audio")
    plt.plot(df["physio_weight"], label="Physio")

    plt.legend()
    plt.title("Modality Contributions")
    plt.xlabel("Time")
    plt.ylabel("Weight")

    plt.show()


def plot_overload(df):
    plt.figure()

    plt.plot(df["overload_probability"])

    plt.title("Overload Probability Over Time")
    plt.xlabel("Time")
    plt.ylabel("Probability")

    plt.show()