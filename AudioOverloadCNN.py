"""AudioOverloadCNN.py

Train a spectrogram-based CNN on the ESC-50 dataset and use it for sensory-overload
risk scoring.

Dataset:
- Clone ESC-50:  git clone https://github.com/karolpiczak/ESC-50.git
- The audio files live in: ESC-50/audio/
- The metadata CSV lives in: ESC-50/meta/esc50.csv

This script:
1) Builds log-mel spectrograms (torch/torchaudio) from ESC-50 audio clips
2) Trains a CNN to classify the 50 ESC-50 categories
3) Converts predicted category probabilities into an 'overload risk score' by mapping
   categories to risk weights (customizable in RISK_WEIGHTS).

Run:
    python AudioOverloadCNN.py --esc50_root /path/to/ESC-50 --epochs 20 --batch_size 32

Inference:
    python AudioOverloadCNN.py --mode predict --model_path artifacts/esc50_cnn.pt --audio_file test.wav
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
except Exception as e:
    raise ImportError(
        "torchaudio is required for this script. Install: pip install torch torchaudio"
    ) from e

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required. Install: pip install pandas") from e


# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Sensory overload risk mapping
# ---------------------------
# ESC-50 categories are things like: "siren", "car_horn", "crow", "rain", "crying_baby", ...
# Here we map category -> risk weight (0.0 to 1.0).
# You should tune these weights to your target population and app logic.
DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    # High-risk / common overload triggers
    "siren": 1.0,
    "car_horn": 1.0,
    "engine": 0.8,
    "train": 0.8,
    "airplane": 0.7,
    "jackhammer": 1.0,
    "drilling": 0.9,
    "chainsaw": 0.9,
    "helicopter": 0.7,
    "church_bells": 0.6,
    "fireworks": 1.0,
    "gun_shot": 1.0,
    "crackling_fire": 0.6,
    "crying_baby": 0.8,
    "crowd": 0.7,
    "clapping": 0.4,
    "laughing": 0.4,
    "sneezing": 0.3,
    "coughing": 0.4,
    "door_wood_knock": 0.4,
    "glass_breaking": 0.9,

    # Natural / typically lower-risk
    "rain": 0.2,
    "sea_waves": 0.2,
    "crickets": 0.2,
    "chirping_birds": 0.2,
    "wind": 0.2,
    "thunderstorm": 0.7,  # can be high for some users

    # Default fallback handled later
}


def risk_score_from_probs(category_probs: torch.Tensor, categories: List[str], risk_weights: Dict[str, float]) -> float:
    """Compute an overload risk score in [0, 1] from predicted category probabilities."""
    # category_probs: (num_classes,)
    score = 0.0
    for i, cat in enumerate(categories):
        w = float(risk_weights.get(cat, 0.5))  # fallback medium risk if unknown
        score += float(category_probs[i]) * w
    return float(max(0.0, min(1.0, score)))


def risk_bucket(score: float) -> str:
    """Convert score into app-friendly buckets."""
    if score < 0.20:
        return "safe"
    if score < 0.40:
        return "moderately_ok"
    if score < 0.60:
        return "dangerous"
    if score < 0.80:
        return "very_dangerous"
    return "extremely_dangerous"


# ---------------------------
# Dataset
# ---------------------------
@dataclass
class AudioSpecConfig:
    sample_rate: int = 44100
    # ESC-50 clips are 5 seconds; we crop/pad to fixed duration for batching.
    clip_seconds: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 20
    f_max: int = 20000


class ESC50SpectrogramDataset(Dataset):
    def __init__(
        self,
        esc50_root: str,
        folds: List[int],
        spec_cfg: AudioSpecConfig,
        normalize: bool = True,
        augment: bool = True,
    ):
        self.esc50_root = esc50_root
        self.audio_dir = os.path.join(esc50_root, "audio")
        self.csv_path = os.path.join(esc50_root, "meta", "esc50.csv")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"ESC-50 metadata not found at: {self.csv_path}")
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"ESC-50 audio dir not found at: {self.audio_dir}")

        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        self.categories = sorted(self.df["category"].unique().tolist())
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}

        self.spec_cfg = spec_cfg
        self.normalize = normalize
        self.augment = augment

        self.mel = MelSpectrogram(
            sample_rate=spec_cfg.sample_rate,
            n_fft=spec_cfg.n_fft,
            hop_length=spec_cfg.hop_length,
            n_mels=spec_cfg.n_mels,
            f_min=spec_cfg.f_min,
            f_max=spec_cfg.f_max,
            power=2.0,
            normalized=False,
        )
        self.db = AmplitudeToDB(stype="power", top_db=80)

        self.target_len = int(spec_cfg.sample_rate * spec_cfg.clip_seconds)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load(filepath)  # (channels, samples)
        if sr != self.spec_cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.spec_cfg.sample_rate)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Crop/pad
        n = wav.size(1)
        if n > self.target_len:
            # random crop during training
            if self.augment:
                start = random.randint(0, n - self.target_len)
            else:
                start = (n - self.target_len) // 2
            wav = wav[:, start : start + self.target_len]
        elif n < self.target_len:
            pad = self.target_len - n
            wav = F.pad(wav, (0, pad))

        # Augment: small gain + noise
        if self.augment:
            if random.random() < 0.5:
                gain = 0.8 + random.random() * 0.4  # [0.8, 1.2]
                wav = wav * gain
            if random.random() < 0.3:
                noise = torch.randn_like(wav) * 0.003
                wav = wav + noise

        return wav

    def _to_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, samples)
        mel = self.mel(wav)  # (1, n_mels, time)
        log_mel = self.db(mel)
        # Normalize per-sample (helps CNN)
        if self.normalize:
            m = log_mel.mean()
            s = log_mel.std().clamp_min(1e-6)
            log_mel = (log_mel - m) / s
        return log_mel

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        fname = row["filename"]
        category = row["category"]
        y = self.cat_to_idx[category]

        path = os.path.join(self.audio_dir, fname)
        wav = self._load_audio(path)
        spec = self._to_log_mel(wav)  # (1, n_mels, time)
        return spec, y


# ---------------------------
# CNN Model
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ESC50CNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += xb.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def train(
    esc50_root: str,
    out_dir: str = "artifacts",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    folds_train: List[int] = [1, 2, 3, 4],
    folds_val: List[int] = [5],
) -> str:
    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec_cfg = AudioSpecConfig()
    ds_train = ESC50SpectrogramDataset(esc50_root, folds_train, spec_cfg, augment=True)
    ds_val = ESC50SpectrogramDataset(esc50_root, folds_val, spec_cfg, augment=False)

    # Ensure consistent category order
    categories = ds_train.categories
    assert categories == ds_val.categories, "Train/val category mismatch."

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ESC50CNN(num_classes=len(categories)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_path = os.path.join(out_dir, "esc50_cnn.pt")

    # Save categories + default risk weights for inference
    meta_path = os.path.join(out_dir, "esc50_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": categories,
                "spec_cfg": spec_cfg.__dict__,
                "risk_weights": DEFAULT_RISK_WEIGHTS,
            },
            f,
            indent=2,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item()) * xb.size(0)
            seen += xb.size(0)

        scheduler.step()
        train_loss = running / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict()}, best_path)

    print(f"Best val_acc={best_acc:.4f} | saved: {best_path}")
    return best_path


# ---------------------------
# Inference
# ---------------------------
def _load_meta(meta_path: str) -> Tuple[List[str], AudioSpecConfig, Dict[str, float]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    categories = meta["categories"]
    spec_cfg = AudioSpecConfig(**meta["spec_cfg"])
    risk_weights = meta.get("risk_weights", DEFAULT_RISK_WEIGHTS)
    return categories, spec_cfg, risk_weights


@torch.no_grad()
def predict_audio(
    model_path: str,
    meta_path: str,
    audio_file: str,
    topk: int = 5,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories, spec_cfg, risk_weights = _load_meta(meta_path)

    model = ESC50CNN(num_classes=len(categories)).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wav, sr = torchaudio.load(audio_file)
    if sr != spec_cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, spec_cfg.sample_rate)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    target_len = int(spec_cfg.sample_rate * spec_cfg.clip_seconds)
    n = wav.size(1)
    if n > target_len:
        start = (n - target_len) // 2
        wav = wav[:, start : start + target_len]
    elif n < target_len:
        wav = F.pad(wav, (0, target_len - n))

    mel = MelSpectrogram(
        sample_rate=spec_cfg.sample_rate,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        f_min=spec_cfg.f_min,
        f_max=spec_cfg.f_max,
        power=2.0,
    )(wav)
    log_mel = AmplitudeToDB(stype="power", top_db=80)(mel)
    log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)

    xb = log_mel.unsqueeze(0).to(device)  # (1, 1, n_mels, time)
    logits = model(xb).squeeze(0)
    probs = torch.softmax(logits, dim=0).cpu()

    # Top-k classes
    top_vals, top_idx = torch.topk(probs, k=min(topk, len(categories)))
    top = [
        {"category": categories[int(i)], "prob": float(p)}
        for p, i in zip(top_vals.tolist(), top_idx.tolist())
    ]

    score = risk_score_from_probs(probs, categories, risk_weights)
    return {
        "top_predictions": top,
        "overload_risk_score": float(score),
        "overload_bucket": risk_bucket(score),
    }


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--esc50_root", type=str, default="ESC-50")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    # predict args
    p.add_argument("--model_path", type=str, default="artifacts/esc50_cnn.pt")
    p.add_argument("--meta_path", type=str, default="artifacts/esc50_meta.json")
    p.add_argument("--audio_file", type=str, default=None)
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(
            esc50_root=args.esc50_root,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    else:
        if not args.audio_file:
            raise ValueError("--audio_file is required in predict mode")
        result = predict_audio(
            model_path=args.model_path,
            meta_path=args.meta_path,
            audio_file=args.audio_file,
            topk=args.topk,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
