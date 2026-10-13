"""
Argus prediction service.

GET /predict?instance_type=m5.xlarge&az=eu-north-1a  ->  {"risk_score": 0.012, ...}

The score is a calibrated probability that the instance's Spot price jumps more than 1% within
the next 15 minutes (the proxy label the model was trained on; see docs/wide_model_result.md).
Features are computed live from recent price updates (PRICE_SOURCE=ec2|s3|csv), not from a frozen
file. The model bundle is ml/model/final/{model.json,meta.json}.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import price_source
from predictor import NoHistory, Predictor

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-north-1"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model", "final"))

predictor = None
load_error = None


def _load():
    global predictor, load_error
    try:
        src = price_source.CachedSource(price_source.from_env(REGION),
                                        ttl_s=int(os.getenv("PRICE_TTL_SECONDS", "300")))
        predictor = Predictor(MODEL_DIR, src, REGION)
        load_error = None
    except Exception as e:  # keep the process up so /health reports why instead of crash-looping
        predictor, load_error = None, f"{type(e).__name__}: {e}"
        print(f"Error loading model: {load_error}")


@asynccontextmanager
async def lifespan(_app):
    _load()
    yield


app = FastAPI(title="Argus Spot Prediction Service", lifespan=lifespan)


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error or "model not loaded")
    return {"status": "ok", "source": predictor.source.name, "region": REGION}


@app.get("/model")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error or "model not loaded")
    m = predictor.meta
    return {"model": m.get("model"), "label": m.get("label"), "features": m["feature_names"],
            "metrics": m.get("metrics"), "thresholds": m.get("thresholds"), "thresholds_by_region": m.get("thresholds_by_region"),
            "instance_types": predictor.instance_types, "region": REGION}


class PredictRequest(BaseModel):
    instance_type: str
    az: str


def _predict(instance_type: str, az: str):
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error or "model not loaded")
    try:
        out = predictor.predict(instance_type, az)
    except NoHistory as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"price source failed: {type(e).__name__}: {e}")
    out["timestamp"] = datetime.now(timezone.utc).isoformat()
    return out


@app.post("/predict")
def predict_post(req: PredictRequest):
    return _predict(req.instance_type, req.az)


@app.get("/predict")
def predict_get(instance_type: str, az: str):
    return _predict(instance_type, az)
