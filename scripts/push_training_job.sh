#!/bin/bash
# Build and push training-job image to ECR.
# Run this manually for Week 6 before CI/CD is live.
# Usage: bash scripts/push_training_job.sh

set -e

REGION="eu-north-1"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argus_env.sh"
ACCOUNT="$ARGUS_ACCOUNT_ID"
REPO="argus/training-job"
TAG="${1:-latest}"
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"

echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "Building training-job image..."
docker build -t "$IMAGE" ml/cifar10_job/

echo "Pushing $IMAGE..."
docker push "$IMAGE"

echo "Done. Image: $IMAGE"
