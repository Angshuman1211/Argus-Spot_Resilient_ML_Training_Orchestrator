#!/bin/bash
# Build and push predict-service image to ECR.
# Run this manually for Week 6 before CI/CD is live.
# Usage: bash scripts/push_predict_service.sh

set -e

REGION="eu-north-1"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argus_env.sh"
ACCOUNT="$ARGUS_ACCOUNT_ID"
REPO="argus/predict-service"
TAG="${1:-latest}"
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"

echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "Building predict-service image from root context..."
# Build from root directory since Dockerfile references ml/ paths
docker build -t "$IMAGE" -f ml/api/Dockerfile .

echo "Pushing $IMAGE..."
docker push "$IMAGE"

echo "Done. Image: $IMAGE"
