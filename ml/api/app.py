"""
Argus prediction service.

GET /predict?instance_type=m5.xlarge&az=eu-north-1a  ->  {"risk_score": 0.012, ...}
GET /reclaim_risk?instance_type=m5.xlarge&az=eu-north-1a&instance_age_min=12  ->  {"reclaim_risk": 0.2, ...}

/predict is a calibrated probability that the instance's Spot price jumps more than 1% within the next
15 minutes (the proxy label the price model was trained on; see docs/wide_model_result.md). Features are
computed live from recent price updates (PRICE_SOURCE=ec2|s3|csv). Bundle: ml/model/final/{model,meta}.json.

/reclaim_risk is the probability AWS reclaims the instance within 15 minutes, from a model trained on real
reclaims (AWS placement score and instance age; capacity.py). The placement score comes from
PLACEMENT_SOURCE=ec2, or from the caller's placement_score parameter. Bundle: capacity_model.json.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import capacity
import price_source
from predictor import NoHistory, Predictor

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-north-1"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model", "final"))

predictor = None
load_error = None
capacity_model = None
placement = None
capacity_error = None


def _load():
    global predictor, load_error, capacity_model, placement, capacity_error
    try:
        src = price_source.CachedSource(price_source.from_env(REGION),
                                        ttl_s=int(os.getenv("PRICE_TTL_SECONDS", "300")))
        predictor = Predictor(MODEL_DIR, src, REGION)
        load_error = None
    except Exception as e:  # keep the process up so /health reports why instead of crash-looping
        predictor, load_error = None, f"{type(e).__name__}: {e}"
        print(f"Error loading model: {load_error}")
    # the capacity model is independent: a broken price feed must not take it down, nor the reverse
    try:
        capacity_model = capacity.CapacityModel.load(MODEL_DIR)
        placement = capacity.placement_source_from_env(REGION)
        capacity_error = None if capacity_model else "no capacity_model.json in the model bundle"
    except Exception as e:
        capacity_model, placement, capacity_error = None, None, f"{type(e).__name__}: {e}"
        print(f"Error loading capacity model: {capacity_error}")


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
            "instance_types": predictor.instance_types, "region": REGION,
            "capacity_model": capacity_model.info() if capacity_model else None}


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


@app.get("/reclaim_risk")
def reclaim_risk(instance_type: str, az: str, instance_age_min: float, placement_score: float | None = None):
    if capacity_model is None:
        raise HTTPException(status_code=503, detail=capacity_error or "capacity model not loaded")
    if instance_age_min < 0:
        raise HTTPException(status_code=422, detail="instance_age_min must be >= 0")
    source = "caller"
    if placement_score is None:
        try:
            placement_score, source = placement.get(instance_type, az), placement.name
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"placement source failed: {type(e).__name__}: {e}")
        if placement_score is None:
            raise HTTPException(status_code=404, detail=f"no placement score for {instance_type} in {az} "
                                                        f"(PLACEMENT_SOURCE={placement.name}; pass placement_score)")
    out = capacity_model.score(placement_score, instance_age_min)
    out.update(instance_type=instance_type, az=az, placement_score=placement_score, placement_source=source,
               instance_age_min=instance_age_min, timestamp=datetime.now(timezone.utc).isoformat())
    return out
