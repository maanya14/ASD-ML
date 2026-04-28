import argparse
import json
import queue
import time
from collections import deque
from typing import Any, Dict, List, Tuple, Optional
import torchaudio 
torchaudio.set_audio_backend("soundfile")

import numpy as np
import sounddevice as sd
import torch
import torchaudio


# ----------------------------
# Official ESC-50 class ordering (50 classes)
# ----------------------------
ESC50_CLASSES_50: List[str] = [
    # Animals (10)
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    # Natural soundscapes & water sounds (10)
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm",
    # Human non-speech sounds (10)
    "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
    "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    # Interior/domestic sounds (10)
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
    # Exterior/urban noises (10)
    "helicopter", "chainsaw", "siren", "car_horn", "engine", "train",
    "church_bells", "airplane", "fireworks", "hand_saw",
]

DEFAULT_RISK_WEIGHTS = {
    # high overload risk
    "siren": 1.0,
    "car_horn": 0.95,
    "fireworks": 0.90,
    "glass_breaking": 0.90,
    "chainsaw": 0.85,
    "crying_baby": 0.85,
    "clock_alarm": 0.80,
    "engine": 0.65,
    "train": 0.65,
    "helicopter": 0.70,
    # medium
    "vacuum_cleaner": 0.60,
    "door_wood_knock": 0.55,
    "mouse_click": 0.45,
    "keyboard_typing": 0.45,
    "clock_tick": 0.40,
    # low
    "rain": 0.20,
    "sea_waves": 0.20,
    "wind": 0.15,
    "chirping_birds": 0.15,
    "crickets": 0.15,
}

# ----------------------------
# Bucketing
# ----------------------------
def bucket_from_score(score: float) -> str:
    if score < 0.20:
        return "safe"
    elif score < 0.40:
        return "moderately_ok"
    elif score < 0.60:
        return "dangerous"
    elif score < 0.80:
        return "very_dangerous"
    else:
        return "extremely_dangerous"


# ----------------------------
# Device selection
# ----------------------------
def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


# ----------------------------
# Checkpoint parsing
# ----------------------------
def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model_state", "state_dict", "model", "weights", "net", "params"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # If it already looks like a state_dict
        tensor_like = True
        for v in ckpt.values():
            if not torch.is_tensor(v):
                tensor_like = False
                break
        if tensor_like:
            return ckpt

    raise RuntimeError(
        "Could not extract a model state_dict from checkpoint. "
        "Your .pt file is not in a supported format."
    )


def infer_num_classes_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Try to infer the number of output classes from the classifier weight shape.
    Looks for a 2D weight tensor where first dim is likely num_classes.
    """
    # common classifier keys first
    for k in ["classifier.4.weight", "fc.weight", "head.weight", "classifier.weight", "linear.weight"]:
        if k in state and state[k].dim() == 2:
            return int(state[k].shape[0])

    # fallback: scan all tensors for a 2D weight with plausible class dim
    candidates = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and k.endswith(".weight"):
            out_dim = int(v.shape[0])
            in_dim = int(v.shape[1])
            # typical class counts are 10/50 etc; accept 5..200 to be safe
            if 5 <= out_dim <= 200 and in_dim >= 8:
                candidates.append((out_dim, k))
    if candidates:
        # pick the largest out_dim (usually the classifier)
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][0]

    return None


# ----------------------------
# Try importing exact training model (best)
# ----------------------------
def try_import_training_model(num_classes: int):
    try:
        import AudioOverloadCNN as train_mod
    except Exception:
        return None

    for fn_name in ["build_model", "get_model", "create_model", "make_model"]:
        fn = getattr(train_mod, fn_name, None)
        if callable(fn):
            try:
                return fn(num_classes)
            except Exception:
                pass

    for cls_name in ["ESC50CNN", "SimpleCNN", "AudioCNN", "CNNModel", "Model"]:
        cls = getattr(train_mod, cls_name, None)
        if isinstance(cls, type):
            try:
                return cls(num_classes)
            except Exception:
                try:
                    return cls(num_classes=num_classes)
                except Exception:
                    pass

    return None


# ----------------------------
# Fallback model (only if import fails)
# ----------------------------
class FallbackCNN(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ----------------------------
# Meta loading (robust)
# ----------------------------
def load_meta(meta_path: str) -> Tuple[List[str], Dict[str, float]]:
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    classes = meta.get("classes") or meta.get("class_names") or meta.get("labels")
    if not isinstance(classes, list) or len(classes) == 0:
        classes = ESC50_CLASSES_50.copy()

    risk_weights = meta.get("risk_weights", {})
    if not isinstance(risk_weights, dict) or len(risk_weights) == 0:
        risk_weights = DEFAULT_RISK_WEIGHTS.copy()

    risk_weights = {k: float(v) for k, v in risk_weights.items()}
    return classes, risk_weights


# ----------------------------
# Audio transforms
# ----------------------------
def build_transform(sr: int, n_fft: int, hop: int, n_mels: int):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return mel, to_db


def audio_to_logmel(wave: np.ndarray, mel, to_db) -> torch.Tensor:
    wav = torch.from_numpy(wave).float().unsqueeze(0)   # [1, T]
    m = mel(wav)                                        # [1, n_mels, frames]
    lm = to_db(m)                                       # log-mel
    lm = (lm - lm.mean()) / (lm.std() + 1e-6)           # normalize
    lm = lm.unsqueeze(0)                                # [B=1, C=1, n_mels, frames]
    return lm


# ----------------------------
# Overload score
# ----------------------------
def compute_overload_score(probs: np.ndarray, classes: List[str], risk_weights: Dict[str, float]) -> float:
    score = 0.0
    for i, c in enumerate(classes):
        w = float(risk_weights.get(c, 0.5))
        score += float(probs[i]) * w
    return float(np.clip(score, 0.0, 1.0))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="artifacts/esc50_cnn.pt")
    ap.add_argument("--meta_path", default="artifacts/esc50_meta.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--hop_seconds", type=float, default=0.75)

    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=64)

    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict")
    args = ap.parse_args()

    device = pick_device(args.device)

    # Load checkpoint and infer output classes from it
    ckpt = torch.load(args.model_path, map_location=device)
    state = extract_state_dict(ckpt)
    inferred_classes = infer_num_classes_from_state(state)

    # Load meta, but override class count if mismatch with checkpoint
    classes, risk_weights = load_meta(args.meta_path)

    if inferred_classes is not None and inferred_classes != len(classes):
        # Most likely your meta json has wrong class list. Override safely.
        if inferred_classes == 50:
            classes = ESC50_CLASSES_50.copy()
        else:
            # If it's some other class count, create placeholder labels.
            classes = [f"class_{i}" for i in range(inferred_classes)]

        print(f"⚠️ Meta classes count mismatch. Using checkpoint output classes = {len(classes)}")

    num_classes = len(classes)

    # Build model (prefer using training model)
    model = try_import_training_model(num_classes)
    if model is None:
        model = FallbackCNN(num_classes)

    model = model.to(device)
    model.load_state_dict(state, strict=bool(args.strict))
    model.eval()

    mel, to_db = build_transform(args.sr, args.n_fft, args.hop_length, args.n_mels)

    window_samples = int(args.seconds * args.sr)
    buffer = deque([0.0] * window_samples, maxlen=window_samples)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        mono = indata[:, 0].astype(np.float32)
        q.put(mono)

    print("\n🎤 Listening... Press Ctrl+C to stop.\n")
    print(f"Device={device} | SR={args.sr} | window={args.seconds}s | step={args.hop_seconds}s | classes={num_classes}\n")

    last_infer_at = 0.0
    smooth_scores = deque(maxlen=6)

    try:
        with sd.InputStream(channels=1, samplerate=args.sr, dtype="float32", callback=callback):
            while True:
                chunk = q.get()
                for x in chunk:
                    buffer.append(float(x))

                now = time.time()
                if now - last_infer_at >= args.hop_seconds:
                    last_infer_at = now

                    wave = np.array(buffer, dtype=np.float32)
                    x = audio_to_logmel(wave, mel, to_db).to(device)

                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    score = compute_overload_score(probs, classes, risk_weights)
                    smooth_scores.append(score)
                    score_s = float(np.mean(smooth_scores))
                    bucket = bucket_from_score(score_s)

                    topk_idx = np.argsort(probs)[::-1][: args.topk]
                    top_preds = ", ".join([f"{classes[i]}:{probs[i]:.2f}" for i in topk_idx])

                    print(f"[{bucket:18}] score={score_s:.2f} | top: {top_preds}")

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print("\nError:", e)
        print("\nQuick fixes:")
        print("1) pip install sounddevice soundfile")
        print("2) macOS: System Settings → Privacy & Security → Microphone → allow Terminal/VS Code")
        print("3) If sounddevice fails: brew install portaudio")


if __name__ == "__main__":
    main()
