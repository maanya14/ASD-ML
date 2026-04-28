"""
AudioOverload.py
----------------
Environmental sound classification + sensory-overload scoring using log-mel spectrograms.

Dataset: ESC-50 (https://github.com/karolpiczak/ESC-50)
Expected layout after cloning:
    ESC-50/
      audio/              # 2000 wav files
      meta/esc50.csv       # metadata with columns: filename, fold, target, category, esc10, src_file, take

This module provides:
- ESC50Dataset: PyTorch dataset that returns (log_mel, target, category)
- SimpleCNN: small CNN for log-mel inputs
- train_esc50: training loop (5-fold split or holdout folds)
- predict_file: classify a single wav file
- overload_score: map class probabilities to overload risk levels (customizable)

Install deps (recommended):
    pip install torch torchaudio numpy pandas scikit-learn tqdm

Usage (example):
    python AudioOverload.py train --esc50_root ./ESC-50 --epochs 20 --folds 1 2 3 --val_fold 4
    python AudioOverload.py predict --model ./models/esc50_cnn.pt --labels ./models/labels.json --wav ./test.wav
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except Exception as e:
    raise RuntimeError(
        "torchaudio is required. Install with: pip install torchaudio"
    ) from e


# -------------------------
# Audio / Spectrogram config
# -------------------------
@dataclass
class SpecConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    f_min: int = 20
    f_max: int = 8000
    # ESC-50 clips are 5s. Keep fixed length for CNN input.
    clip_seconds: float = 5.0


# -------------------------
# Dataset
# -------------------------
class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        esc50_root: str,
        folds: List[int],
        spec_cfg: SpecConfig = SpecConfig(),
        train: bool = True,
        augment: bool = True,
    ) -> None:
        self.esc50_root = esc50_root
        self.audio_dir = os.path.join(esc50_root, "audio")
        self.meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"ESC-50 meta not found at {self.meta_path}. "
                f"Clone the dataset repo first: git clone https://github.com/karolpiczak/ESC-50.git"
            )
        self.spec_cfg = spec_cfg
        self.train = train
        self.augment = augment and train

        df = pd.read_csv(self.meta_path)
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.df = df

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=spec_cfg.sample_rate,
            n_fft=spec_cfg.n_fft,
            hop_length=spec_cfg.hop_length,
            n_mels=spec_cfg.n_mels,
            f_min=spec_cfg.f_min,
            f_max=spec_cfg.f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.target_to_category = (
            df[["target", "category"]]
            .drop_duplicates()
            .sort_values("target")
            .set_index("target")["category"]
            .to_dict()
        )

        # Precompute fixed length
        self.num_samples = int(spec_cfg.sample_rate * spec_cfg.clip_seconds)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)  # (channels, n)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != self.spec_cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.spec_cfg.sample_rate)

        # Pad/trim to fixed length
        n = wav.shape[1]
        if n < self.num_samples:
            pad = self.num_samples - n
            wav = F.pad(wav, (0, pad))
        elif n > self.num_samples:
            wav = wav[:, : self.num_samples]
        return wav

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        # (1, n_mels, time)
        mel = self.mel(wav)
        log_mel = self.to_db(mel)
        # Normalize per-example (helps generalization)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        # Light augmentations that preserve label
        if not self.augment:
            return wav
        # Random gain
        gain_db = float(np.random.uniform(-6, 6))
        wav = wav * (10 ** (gain_db / 20.0))
        # Random time shift (circular)
        shift = int(np.random.uniform(-0.1, 0.1) * wav.shape[1])
        if shift != 0:
            wav = torch.roll(wav, shifts=shift, dims=1)
        # Add small gaussian noise
        noise = torch.randn_like(wav) * float(np.random.uniform(0.0, 0.01))
        wav = wav + noise
        return wav.clamp(-1.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["filename"])
        wav = self._load_audio(wav_path)
        wav = self._augment(wav)
        x = self._spec(wav)  # (1, n_mels, time)
        y = int(row["target"])
        cat = str(row["category"])
        return x, y, cat


# -------------------------
# Model
# -------------------------
class SimpleCNN(nn.Module):
    """
    Small CNN that works well as a baseline for ESC-50.
    Input shape: (B, 1, n_mels, time_frames)
    """
    def __init__(self, num_classes: int = 50) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)
        return self.fc(x)


# -------------------------
# Overload scoring
# -------------------------
DEFAULT_OVERLOAD_WEIGHTS: Dict[str, float] = {
    # 0.0 = likely calm, 1.0 = likely overload
    # Tune these based on your target population / experiments.
    # High-risk (sharp, unpredictable, alarm-like)
    "siren": 1.0,
    "car_horn": 0.9,
    "chainsaw": 0.85,
    "jackhammer": 0.9,
    "engine": 0.7,
    "helicopter": 0.8,
    "gun_shot": 1.0,
    "explosion": 1.0,
    # Crowds / human noise (often stressful)
    "crow": 0.7,      # note: ESC label is "crow" (bird) vs "crowd" not in ESC-50
    "crying_baby": 0.9,
    "laughing": 0.6,
    # Natural / steady sounds (usually lower overload)
    "rain": 0.2,
    "sea_waves": 0.2,
    "crickets": 0.2,
    "wind": 0.25,
}

def overload_score(
    probs_by_category: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> Tuple[float, str]:
    """
    Convert class probabilities into a single overload score in [0, 1].

    If a category isn't listed in weights, a default 0.5 is used.
    """
    weights = weights or DEFAULT_OVERLOAD_WEIGHTS
    score = 0.0
    for cat, p in probs_by_category.items():
        w = float(weights.get(cat, 0.5))
        score += w * float(p)
    score = float(min(max(score, 0.0), 1.0))

    if score < 0.2:
        level = "ok"
    elif score < 0.4:
        level = "moderately_ok"
    elif score < 0.6:
        level = "dangerous"
    elif score < 0.8:
        level = "very_dangerous"
    else:
        level = "extremely_dangerous"
    return score, level


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_esc50(
    esc50_root: str,
    train_folds: List[int],
    val_fold: int,
    out_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> Tuple[str, str]:
    """
    Train a baseline CNN on ESC-50.
    Saves:
      - model .pt
      - labels.json (target->category)
    Returns paths: (model_path, labels_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    spec_cfg = SpecConfig()
    ds_train = ESC50Dataset(esc50_root, folds=train_folds, spec_cfg=spec_cfg, train=True, augment=True)
    ds_val = ESC50Dataset(esc50_root, folds=[val_fold], spec_cfg=spec_cfg, train=False, augment=False)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=50).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = os.path.join(out_dir, "esc50_cnn_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item())

        val_acc = evaluate(model, val_loader, dev)
        avg_loss = running / max(len(train_loader), 1)
        print(f"epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "spec_cfg": spec_cfg.__dict__,
                    "train_folds": train_folds,
                    "val_fold": val_fold,
                },
                best_path,
            )

    labels_path = os.path.join(out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ds_train.target_to_category.items()}, f, indent=2)

    print(f"Saved best model to: {best_path} (val_acc={best_acc:.4f})")
    print(f"Saved labels to:     {labels_path}")
    return best_path, labels_path


# -------------------------
# Inference helpers
# -------------------------
def _load_model(model_path: str, device: str | None = None) -> Tuple[SimpleCNN, SpecConfig, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_path, map_location=dev)
    spec_cfg = SpecConfig(**ckpt.get("spec_cfg", {}))
    model = SimpleCNN(num_classes=50).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, spec_cfg, dev


@torch.no_grad()
def predict_file(
    model_path: str,
    labels_path: str,
    wav_path: str,
    topk: int = 5,
    device: str | None = None,
) -> Dict:
    model, spec_cfg, dev = _load_model(model_path, device=device)

    with open(labels_path, "r", encoding="utf-8") as f:
        target_to_category = {int(k): v for k, v in json.load(f).items()}

    # load wav
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != spec_cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, spec_cfg.sample_rate)

    num_samples = int(spec_cfg.sample_rate * spec_cfg.clip_seconds)
    if wav.shape[1] < num_samples:
        wav = F.pad(wav, (0, num_samples - wav.shape[1]))
    else:
        wav = wav[:, :num_samples]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=spec_cfg.sample_rate,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
        f_min=spec_cfg.f_min,
        f_max=spec_cfg.f_max,
        power=2.0,
    )(wav)
    log_mel = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    x = log_mel.unsqueeze(0).to(dev)  # (1,1,n_mels,time)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)

    # map to category probs
    probs_by_category = {target_to_category[i]: float(probs[i]) for i in range(len(probs))}
    score, level = overload_score(probs_by_category)

    top_idx = probs.argsort()[::-1][:topk]
    top = [{"target": int(i), "category": target_to_category[int(i)], "prob": float(probs[int(i)])} for i in top_idx]

    return {
        "wav": wav_path,
        "top": top,
        "overload_score": score,
        "overload_level": level,
    }


# -------------------------
# CLI
# -------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--esc50_root", required=True, help="Path to ESC-50 repo root (contains audio/ and meta/)")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--out_dir", default="models")
    t.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4], help="Training folds")
    t.add_argument("--val_fold", type=int, default=5, help="Validation fold")
    t.add_argument("--device", default=None)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--labels", required=True)
    pr.add_argument("--wav", required=True)
    pr.add_argument("--topk", type=int, default=5)
    pr.add_argument("--device", default=None)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "train":
        train_esc50(
            esc50_root=args.esc50_root,
            train_folds=args.folds,
            val_fold=args.val_fold,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.cmd == "predict":
        out = predict_file(
            model_path=args.model,
            labels_path=args.labels,
            wav_path=args.wav,
            topk=args.topk,
            device=args.device,
        )
        print(json.dumps(out, indent=2))
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
