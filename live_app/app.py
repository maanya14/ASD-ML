from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
PHYSIO_RESULTS = ROOT / "PHYSIO" / "WESAD_OVERLOAD_RESULTS.csv"
PHYSIO_VALIDATION = ROOT / "PHYSIO" / "physio_validation_output.csv"


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return lo
    return float(max(lo, min(hi, value)))


def _smooth(previous: float, value: float, alpha: float = 0.35) -> float:
    return _clip((previous * (1.0 - alpha)) + (value * alpha))


def _level(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.5:
        return "Moderate"
    if probability < 0.75:
        return "High"
    return "Severe"


def _load_physio() -> pd.DataFrame:
    if PHYSIO_RESULTS.exists():
        df = pd.read_csv(PHYSIO_RESULTS)
        keep = [
            "subject_id",
            "center_time_sec",
            "overload_probability",
            "overload_level",
            "label",
        ]
        return df[keep].rename(columns={"overload_probability": "physio_stress_prob"})

    if PHYSIO_VALIDATION.exists():
        df = pd.read_csv(PHYSIO_VALIDATION)
        keep = ["subject_id", "center_time_sec", "physio_stress_prob", "label"]
        out = df[keep].copy()
        out["overload_level"] = out["physio_stress_prob"].map(_level)
        return out

    return pd.DataFrame(
        {
            "subject_id": ["demo"],
            "center_time_sec": [0.0],
            "physio_stress_prob": [0.35],
            "overload_level": ["Moderate"],
            "label": [0],
        }
    )


PHYSIO_DF = _load_physio()


class AudioFeatures(BaseModel):
    rms: float = Field(ge=0.0, le=1.0)
    peak: float = Field(ge=0.0, le=1.0)
    zero_crossing_rate: float = Field(ge=0.0, le=1.0)
    spectral_centroid: float = Field(ge=0.0, le=1.0)


class VideoFeatures(BaseModel):
    brightness: float = Field(ge=0.0, le=255.0)
    contrast: float = Field(ge=0.0, le=128.0)
    motion: float = Field(ge=0.0, le=255.0)
    flicker: float = Field(ge=0.0, le=255.0)
    face_presence: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    session_id: str
    audio: AudioFeatures
    video: VideoFeatures
    timestamp_ms: int


class SessionCreateRequest(BaseModel):
    subject_id: Optional[str] = None


class ModalityScore(BaseModel):
    name: Literal["audio", "video", "physio"]
    probability: float
    level: str
    signal: Dict[str, Union[float, str]]


class AnalyzeResponse(BaseModel):
    session_id: str
    fusion_probability: float
    fusion_level: str
    confidence: float
    weights: Dict[str, float]
    modalities: List[ModalityScore]
    timeline: List[Dict[str, float]]
    physio_index: int


@dataclass
class SessionState:
    subject_id: Optional[str] = None
    physio_index: int = 0
    samples_seen: int = 0
    audio_score: float = 0.0
    video_score: float = 0.0
    physio_score: float = 0.0
    fusion_score: float = 0.0
    last_seen: float = field(default_factory=time.time)
    timeline: List[Dict[str, float]] = field(default_factory=list)


SESSIONS: Dict[str, SessionState] = {}


def _physio_rows_for_subject(subject_id: Optional[str]) -> pd.DataFrame:
    if subject_id:
        rows = PHYSIO_DF[PHYSIO_DF["subject_id"].astype(str) == str(subject_id)]
        if not rows.empty:
            return rows.reset_index(drop=True)
    return PHYSIO_DF.reset_index(drop=True)


def _next_physio(state: SessionState) -> Dict[str, float | str]:
    rows = _physio_rows_for_subject(state.subject_id)
    if rows.empty:
        rows = PHYSIO_DF.reset_index(drop=True)

    row = rows.iloc[state.physio_index % len(rows)]
    state.physio_index += 1
    prob = _clip(float(row.get("physio_stress_prob", 0.0)))
    return {
        "probability": prob,
        "level": str(row.get("overload_level", _level(prob))),
        "subject_id": str(row.get("subject_id", "demo")),
        "center_time_sec": float(row.get("center_time_sec", 0.0)),
        "label": float(row.get("label", 0.0)),
    }


def _audio_probability(audio: AudioFeatures) -> float:
    loudness = min(audio.rms / 0.16, 1.0)
    transient = min(audio.peak / 0.55, 1.0)
    noisiness = audio.zero_crossing_rate
    brightness = audio.spectral_centroid
    return _clip((0.44 * loudness) + (0.24 * transient) + (0.18 * noisiness) + (0.14 * brightness))


def _video_probability(video: VideoFeatures) -> float:
    bright = abs(video.brightness - 120.0) / 120.0
    contrast = video.contrast / 78.0
    motion = video.motion / 42.0
    flicker = video.flicker / 35.0
    face = video.face_presence * 0.35
    return _clip((0.18 * bright) + (0.22 * contrast) + (0.30 * motion) + (0.25 * flicker) + face)


def _fusion(audio: float, video: float, physio: float) -> Tuple[float, Dict[str, float], float]:
    raw = np.array([audio, video, physio], dtype=float)
    centered = raw - raw.mean()
    weights = np.exp(2.2 * centered)
    weights = weights / weights.sum()
    conservative = 0.18 * max(raw)
    fused = float(np.dot(raw, weights) * 0.82 + conservative)
    confidence = float(1.0 - min(np.std(raw), 0.5))
    return _clip(fused), {
        "audio": float(weights[0]),
        "video": float(weights[1]),
        "physio": float(weights[2]),
    }, _clip(confidence)


app = FastAPI(title="ASD-ML Live Multimodal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "physio_rows": int(len(PHYSIO_DF)),
        "subjects": sorted(PHYSIO_DF["subject_id"].astype(str).unique().tolist()),
    }


@app.post("/api/session")
def create_session(payload: SessionCreateRequest) -> Dict[str, object]:
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = SessionState(subject_id=payload.subject_id)
    return {
        "session_id": session_id,
        "physio_rows": int(len(_physio_rows_for_subject(payload.subject_id))),
        "subjects": sorted(PHYSIO_DF["subject_id"].astype(str).unique().tolist()),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    state = SESSIONS.get(payload.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    physio = _next_physio(state)
    audio_now = _audio_probability(payload.audio)
    video_now = _video_probability(payload.video)

    if state.samples_seen == 0:
        state.audio_score = audio_now
        state.video_score = video_now
        state.physio_score = float(physio["probability"])
    else:
        state.audio_score = _smooth(state.audio_score, audio_now)
        state.video_score = _smooth(state.video_score, video_now)
        state.physio_score = _smooth(state.physio_score, float(physio["probability"]))

    fusion_now, weights, confidence = _fusion(
        state.audio_score,
        state.video_score,
        state.physio_score,
    )
    state.fusion_score = fusion_now if state.samples_seen == 0 else _smooth(state.fusion_score, fusion_now, alpha=0.45)
    state.samples_seen += 1
    state.last_seen = time.time()

    state.timeline.append(
        {
            "t": float(payload.timestamp_ms),
            "audio": state.audio_score,
            "video": state.video_score,
            "physio": state.physio_score,
            "fusion": state.fusion_score,
        }
    )
    state.timeline = state.timeline[-80:]

    return AnalyzeResponse(
        session_id=payload.session_id,
        fusion_probability=state.fusion_score,
        fusion_level=_level(state.fusion_score),
        confidence=confidence,
        weights=weights,
        modalities=[
            ModalityScore(
                name="audio",
                probability=state.audio_score,
                level=_level(state.audio_score),
                signal=payload.audio.dict(),
            ),
            ModalityScore(
                name="video",
                probability=state.video_score,
                level=_level(state.video_score),
                signal=payload.video.dict(),
            ),
            ModalityScore(
                name="physio",
                probability=state.physio_score,
                level=_level(state.physio_score),
                signal={
                    "subject_id": str(physio["subject_id"]),
                    "center_time_sec": float(physio["center_time_sec"]),
                    "label": float(physio["label"]),
                },
            ),
        ],
        timeline=state.timeline,
        physio_index=state.physio_index,
    )
