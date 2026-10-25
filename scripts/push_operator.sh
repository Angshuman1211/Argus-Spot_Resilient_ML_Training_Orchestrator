#!/bin/bash
# Build and push operator image to ECR.
# Run this manually for Week 6 before CI/CD is live.
# Usage: bash scripts/push_operator.sh

set -e

REGION="eu-north-1"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argus_env.sh"
ACCOUNT="$ARGUS_ACCOUNT_ID"
REPO="argus/operator"
TAG="${1:-latest}"
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"

echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "Building operator image..."
docker build -t "$IMAGE" operator/

echo "Pushing $IMAGE..."
docker push "$IMAGE"

echo "Done. Image: $IMAGE"
