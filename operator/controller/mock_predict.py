"""
Mock FastAPI prediction service for local operator testing.

Runs independently of Person B's real service. Returns a configurable
risk score so you can test the full operator control loop without
needing the real ML model running.

Usage:
    pip install fastapi uvicorn
    MOCK_RISK_SCORE=0.9 uvicorn controller.mock_predict:app --port 8000

Then set PREDICT_SERVICE_URL=http://localhost:8000 in .env.local.
To test high-risk path: MOCK_RISK_SCORE=0.9 (above default threshold 0.65)
To test low-risk path:  MOCK_RISK_SCORE=0.3

Modes (MOCK_RISK_MODE):
    constant (default) — always returns MOCK_RISK_SCORE.
    ramp               — risk climbs MOCK_RISK_START -> MOCK_RISK_END over
                         MOCK_RAMP_SECONDS, then holds. Used for the Phase-7
                         Grafana demo: the operator sees risk cross the 0.65
                         threshold and fires a proactive checkpoint, so the
                         dashboard shows the climb -> checkpoint story.
"""

import datetime
import math
import os
import time

from fastapi import FastAPI

app = FastAPI(title="Argus Mock Predict Service")

MOCK_RISK_MODE = os.environ.get("MOCK_RISK_MODE", "constant")
MOCK_RISK_SCORE = float(os.environ.get("MOCK_RISK_SCORE", "0.9"))
MOCK_RISK_START = float(os.environ.get("MOCK_RISK_START", "0.15"))
MOCK_RISK_END = float(os.environ.get("MOCK_RISK_END", "0.90"))
MOCK_RAMP_SECONDS = float(os.environ.get("MOCK_RAMP_SECONDS", "120"))

_START_TS = time.monotonic()


def _current_risk() -> float:
    if MOCK_RISK_MODE == "ramp":
        frac = min(1.0, (time.monotonic() - _START_TS) / MOCK_RAMP_SECONDS)
        return round(MOCK_RISK_START + (MOCK_RISK_END - MOCK_RISK_START) * frac, 4)
    if MOCK_RISK_MODE == "oscillate":
        # sine between start and end — never checkpoints if you keep it under threshold
        t = time.monotonic() - _START_TS
        mid = (MOCK_RISK_START + MOCK_RISK_END) / 2
        amp = (MOCK_RISK_END - MOCK_RISK_START) / 2
        return round(mid + amp * math.sin(t / MOCK_RAMP_SECONDS * 2 * math.pi), 4)
    return MOCK_RISK_SCORE


@app.get("/predict")
def predict(instance_type: str = "m5.large", az: str = "eu-north-1a"):
    return {
        "instance_type": instance_type,
        "az": az,
        "risk_score": _current_risk(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/reclaim_risk")
def reclaim_risk(instance_type: str = "m5.large", az: str = "eu-north-1a", instance_age_min: float = 0.0):
    """Same mock score, shaped like the real service's capacity endpoint (riskSignal: capacity)."""
    return {
        "instance_type": instance_type,
        "az": az,
        "instance_age_min": instance_age_min,
        "reclaim_risk": _current_risk(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
