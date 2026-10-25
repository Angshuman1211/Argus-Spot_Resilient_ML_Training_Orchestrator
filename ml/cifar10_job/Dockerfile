FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
# CPU-only torch keeps the image small (the default Linux wheel pulls CUDA libraries)
RUN pip install --no-cache-dir --default-timeout 120 --retries 10 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout 120 --retries 10 boto3

COPY train.py .

# S3_BUCKET (argus-checkpoints-<account id>) comes from the pod manifest, so the image is account-neutral
ENV S3_PREFIX=checkpoints/cifar10-test/latest_checkpoint.pt
ENV EPOCHS=5

CMD ["python", "train.py"]
