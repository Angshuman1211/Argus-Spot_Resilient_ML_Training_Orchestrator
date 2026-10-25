FROM python:3.11-slim-bookworm

WORKDIR /app

COPY ml/api/requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
RUN pip install --no-cache-dir --default-timeout 120 --retries 10 -r requirements.txt

# Shared feature code + the trained model bundle (written by ml/model/train_final.py).
# No torch, scikit-learn, or baked-in features.csv: prices are fetched live (PRICE_SOURCE).
COPY ml/model/spot_features.py /app/model/spot_features.py
COPY ml/model/final /app/model/final
COPY ml/api/app.py ml/api/predictor.py ml/api/price_source.py ml/api/capacity.py /app/api/

ENV MODEL_DIR=/app/model/final
# PRICE_SOURCE=ec2 (default, needs ec2:DescribeSpotPriceHistory), s3 (needs FEATURE_STORE_BUCKET),
# or csv (needs PRICE_CSV, e.g. a snapshot mounted or baked in for a demo).
ENV PRICE_SOURCE=ec2
# /reclaim_risk: PLACEMENT_SOURCE=ec2 (needs ec2:GetSpotPlacementScores, ec2:DescribeAvailabilityZones)
# or none (the caller passes placement_score).
ENV PLACEMENT_SOURCE=none

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app/api", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
