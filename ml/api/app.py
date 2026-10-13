from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import boto3
import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# Add the model directory to sys.path so we can import the model class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
from transformer import SpotInterruptionPredictor
from feature_config import FEATURE_COLUMNS, SEQ_LENGTH

@asynccontextmanager
async def lifespan(_app):
    _load_model()
    _load_features()
    yield


app = FastAPI(title="Argus Spot Prediction Service", lifespan=lifespan)

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "argus-models")
MODEL_KEY = os.getenv("MODEL_KEY", "spot_transformer.pt")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", None)  # Use http://host.docker.internal:4566 for localstack
FEATURES_CSV = os.getenv("FEATURES_CSV", "/app/data/features.csv")

# Global state
model = None
scaler = None
calibrator = None  # optional: Platt-scaling calibrator, applied to the raw sigmoid output
feature_cache = None
is_ready = False

num_features = len(FEATURE_COLUMNS)
seq_len = SEQ_LENGTH

def _load_model():
    global model, scaler, calibrator, is_ready
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "../model")
        model_path = os.path.join(model_dir, "spot_transformer.pt")
        scaler_path = os.path.join(model_dir, "spot_scaler.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        calibrator_path = os.path.join(model_dir, "spot_calibrator.joblib")

        if not os.path.exists(model_path):
            # In a real environment, you'd download the model from S3 on startup
            print(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY}")
            model_path = "/tmp/model.pt"
            s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT) if S3_ENDPOINT else boto3.client('s3')
            s3.download_file(S3_BUCKET, MODEL_KEY, model_path)
        else:
            print(f"Loading model from local path: {model_path}")

        # The scaler and architecture config are produced by train.py alongside the
        # checkpoint. Without the scaler, serving would feed the model raw (unscaled)
        # values while it was trained on standardized ones — that mismatch is what
        # caused every prediction to collapse to a flat ~base-rate score.
        with open(metadata_path) as f:
            metadata = json.load(f)
        scaler = joblib.load(scaler_path)

        if metadata["feature_columns"] != FEATURE_COLUMNS or metadata["seq_length"] != SEQ_LENGTH:
            raise ValueError(
                "model_metadata.json does not match feature_config.py — "
                "retrain the model so serving and training agree on the feature contract."
            )

        model = SpotInterruptionPredictor(
            num_features=metadata["num_features"],
            d_model=metadata["d_model"],
            nhead=metadata["nhead"],
            num_layers=metadata["num_layers"],
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Optional: raw model output is over-confident (see calibrate_and_finalize.py -
        # uncalibrated Brier score is worse than a trivial "always predict safe"
        # baseline). If a calibrator was fit, apply it; if not, fall back to the raw
        # sigmoid rather than failing startup over an optional artifact.
        if os.path.exists(calibrator_path):
            calibrator = joblib.load(calibrator_path)
            print("Calibrator loaded — risk_score will be Platt-scaled.")
        elif os.getenv("ALLOW_UNCALIBRATED") == "1":
            calibrator = None
            print("No calibrator found - risk_score will be the raw (uncalibrated) sigmoid output.")
        else:
            # Operator thresholds are set on the calibrated scale (~0.0015). Serving raw
            # scores would trip them on nearly every poll, so refuse to start instead.
            raise FileNotFoundError(
                f"{calibrator_path} missing. Set ALLOW_UNCALIBRATED=1 to serve raw scores (tests only)."
            )

        is_ready = True
        print("Model, scaler, and metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        is_ready = False

def _load_features():
    global feature_cache
    try:
        local_features_path = os.path.join(os.path.dirname(__file__), "../data/features.csv")
        path_to_use = local_features_path if os.path.exists(local_features_path) else FEATURES_CSV

        df = pd.read_csv(path_to_use)

        cache = {}
        for (instance_type, az), group in df.groupby(['instance_type', 'availability_zone']):
            # Sort by timestamp to get the latest seq_len rows, same column set/order
            # (and same fillna(0)) training used, so the scaler transform is valid.
            sorted_group = group.sort_values('timestamp')
            recent_data = sorted_group[FEATURE_COLUMNS].fillna(0).values[-seq_len:]

            # Scale BEFORE padding: a raw 0 means nothing to the model (it never saw raw
            # values), but a post-scale 0 is the training-set mean per feature — a
            # reasonable neutral filler for instance/AZ pairs with too little history.
            scaled = scaler.transform(recent_data)
            if len(scaled) < seq_len:
                pad_size = seq_len - len(scaled)
                padding = np.zeros((pad_size, num_features))
                scaled = np.vstack([padding, scaled])

            cache[(instance_type, az)] = scaled

        feature_cache = cache
        print(f"Loaded features for {len(feature_cache)} (instance_type, az) combinations.")
    except Exception as e:
        print(f"Error loading features: {e}")
        feature_cache = {}

@app.get("/health")
def health_check():
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ok"}

class PredictRequest(BaseModel):
    instance_type: str
    az: str

@app.post("/predict")
def predict_post(req: PredictRequest):
    return _predict(req.instance_type, req.az)

@app.get("/predict")
def predict_get(instance_type: str, az: str):
    return _predict(instance_type, az)

def _predict(instance_type: str, az: str):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    key = (instance_type, az)
    if key not in feature_cache:
        raise HTTPException(status_code=404, detail="No historical features found for this instance type & AZ")
        
    # Already scaled with the same StandardScaler fit during training (see
    # _load_features). Serving used to feed the model raw values while it was trained
    # on standardized ones — that mismatch is what made every prediction collapse to a
    # flat, near-constant score.
    recent_features = feature_cache[key]

    # Convert to tensor [batch=1, seq_len, num_features]
    x_tensor = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x_tensor)
        raw_prob = torch.sigmoid(logits).item()

    if calibrator is not None:
        eps = 1e-7
        clipped = min(max(raw_prob, eps), 1 - eps)
        logit = np.log(clipped / (1 - clipped))
        risk_score = float(calibrator.predict_proba([[logit]])[0, 1])
    else:
        risk_score = raw_prob

    return {
        "instance_type": instance_type,
        "az": az,
        "risk_score": round(risk_score, 4),
        "calibrated": calibrator is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
