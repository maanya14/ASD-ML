import os
import av
import cv2
import numpy as np
import pandas as pd

WINDOW = 5
STEP = 2

def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def analyze_video(video_path, video_id):
    container = av.open(video_path)
    stream = next((s for s in container.streams if s.type == "video"), None)

    fps = float(stream.average_rate) if stream.average_rate else 24.0

    brightness, motion, contrast, flicker = [], [], [], []
    prev_gray = None
    prev_brightness = None

    for frame in container.decode(stream):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b = np.mean(gray)
        c = np.std(gray)

        if prev_gray is None:
            m = 0
        else:
            m = np.mean(cv2.absdiff(gray, prev_gray))

        if prev_brightness is None:
            f = 0
        else:
            f = abs(b - prev_brightness)

        brightness.append(b)
        contrast.append(c)
        motion.append(m)
        flicker.append(f)

        prev_gray = gray
        prev_brightness = b

    total_secs = int(len(brightness) / fps)
    win_frames = int(WINDOW * fps)

    rows = []

    for start in range(0, total_secs - WINDOW + 1, STEP):
        sf = int(start * fps)
        ef = sf + win_frames

        rows.append({
            "video_id": video_id,
            "timestamp": to_timestamp(start),
            "brightness": np.mean(brightness[sf:ef]),
            "contrast": np.mean(contrast[sf:ef]),
            "motion": np.mean(motion[sf:ef]),
            "flicker": np.mean(flicker[sf:ef])
        })

    return rows