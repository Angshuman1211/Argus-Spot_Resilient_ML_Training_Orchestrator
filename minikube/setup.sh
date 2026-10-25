#!/usr/bin/env bash
# Minikube + LocalStack dev environment setup for Argus operator.
# Run once after `minikube start`.
# Re-running is safe — kubectl apply is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. LocalStack
# ---------------------------------------------------------------------------
echo "==> Starting LocalStack..."
docker compose -f "$REPO_ROOT/localstack/docker-compose.yml" up -d

echo "==> Waiting for LocalStack to be ready..."
until curl -s http://localhost:4566/_localstack/health | grep -q '"s3": "available"'; do
  sleep 2
done
echo "    LocalStack is up."

echo "==> Verifying LocalStack buckets and queues..."
aws --endpoint-url http://localhost:4566 s3 ls
aws --endpoint-url http://localhost:4566 sqs list-queues

# ---------------------------------------------------------------------------
# 2. Minikube — CRD + RBAC
# ---------------------------------------------------------------------------
echo ""
echo "==> Applying CRD..."
kubectl apply -f "$REPO_ROOT/operator/crd/spotresilientjob.yaml"

echo "==> Applying RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding)..."
envsubst < "$REPO_ROOT/operator/rbac.yaml" | kubectl apply -f -  # IRSA ARN unused off-EKS

echo "==> Verifying CRD is registered..."
kubectl get crd spotresilientjobs.argus.io

# ---------------------------------------------------------------------------
# 3. Instructions
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Local dev environment ready."
echo "========================================================"
echo ""
echo "Run the operator (pointed at LocalStack):"
echo "  cd $REPO_ROOT/operator"
echo "  source .env.local"
echo "  kopf run controller/handlers.py --verbose"
echo ""
echo "Submit a test job:"
echo "  envsubst < $REPO_ROOT/demo/spotresilientjob.yaml | kubectl apply -f -"
echo "  kubectl get spotresilientjobs"
echo ""
echo "Tear down when done:"
echo "  minikube stop"
echo "  docker compose -f $REPO_ROOT/localstack/docker-compose.yml down"
