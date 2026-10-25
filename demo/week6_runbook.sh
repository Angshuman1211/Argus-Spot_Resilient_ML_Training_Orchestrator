#!/usr/bin/env bash
#
# Week 6 — real-EKS interruption test, cost-bounded.
#
# The EKS control plane bills $0.10/hr and Spot nodes bill per-second the whole
# time they run. This script keeps the billed window as short as possible by
# splitting the work into explicit phases you run one at a time:
#
#   ./week6_runbook.sh up       # apply node groups (STARTS THE METER)
#   ./week6_runbook.sh deploy   # push operator image + helm install + apply job
#   ./week6_runbook.sh test     # force high risk, watch the migration chain
#   ./week6_runbook.sh status   # nodes / pods / operator logs at a glance
#   ./week6_runbook.sh down     # destroy node groups (STOPS THE METER) <-- ALWAYS RUN
#
# `up` and `down` are independent — `down` tears down whatever is running, even
# if `up`/`deploy` half-failed. When in doubt, run `down`.
#
set -euo pipefail

# ---- resolve paths relative to this script, not the caller's cwd ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TF_DIR="$PROJECT_ROOT/terraform"

# ---- config (matches helm/argus/values.yaml + terraform) ----
REGION="eu-north-1"
CLUSTER="argus-eks"
. "$SCRIPT_DIR/../scripts/argus_env.sh"
REGISTRY="$ARGUS_REGISTRY"
OPERATOR_REPO="argus/operator"
CHECKPOINT_BUCKET="$ARGUS_CKPT_BUCKET"
HELM_RELEASE="argus"
NAMESPACE="default"
SPOT_NODES=2   # 2 needed: reschedule deletes the pod so it must land on a *second* node

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }

confirm() {
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "aborted."; exit 1; }
}

cmd_up() {
  warn "This starts real Spot EC2 billing. Run 'down' the moment the test is over."
  confirm "Bring up $SPOT_NODES Spot node(s) + system node on real EKS?"
  log "Applying node groups (control plane assumed already live for IRSA)"
  terraform -chdir="$TF_DIR" apply \
    -var="spot_desired_size=$SPOT_NODES" \
    -target=aws_eks_node_group.system_nodes \
    -target=aws_eks_node_group.spot_nodes
  log "Wiring kubeconfig to $CLUSTER"
  aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"
  log "Waiting for nodes to register..."
  kubectl wait --for=condition=Ready nodes --all --timeout=300s
  kubectl get nodes -L lifecycle,workload
}

cmd_deploy() {
  log "Logging in to ECR + pushing operator image"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
  # --platform linux/amd64: the EKS nodes are amd64; an Apple-Silicon build
  # host would otherwise produce an arm64 image → "exec format error" crash.
  docker build --platform linux/amd64 -t "$REGISTRY/$OPERATOR_REPO:latest" "$PROJECT_ROOT/operator/"
  docker push "$REGISTRY/$OPERATOR_REPO:latest"

  warn "Person B must have pushed argus/predict-service + argus/training-job to ECR."
  warn "predict-service should run the REAL model here, not MOCK_MODE."
  confirm "Are Person B's images in ECR?"

  log "Deploying operator via Helm"
  helm upgrade --install "$HELM_RELEASE" "$PROJECT_ROOT/helm/argus" \
    --namespace "$NAMESPACE" --wait --timeout 5m

  log "Deploying predict-service + the SpotResilientJob"
  envsubst < "$PROJECT_ROOT/k8s/predict-service.yaml" | kubectl apply -f -
  envsubst < "$PROJECT_ROOT/demo/spotresilientjob.yaml" | kubectl apply -f -
  kubectl get pods -w &
  WATCH_PID=$!
  sleep 20 && kill "$WATCH_PID" 2>/dev/null || true
}

cmd_test() {
  # Controlled test: force the operator over its risk threshold without waiting
  # for a real Spot signal. Lower the threshold so the current risk score trips it.
  log "Forcing high-risk: setting operator RISK_THRESHOLD=0.1"
  helm upgrade "$HELM_RELEASE" "$PROJECT_ROOT/helm/argus" \
    --namespace "$NAMESPACE" --reuse-values \
    --set operator.riskThreshold=0.1 --wait

  log "Watch the migration chain — expect, in order:"
  echo "  1. operator logs: risk > threshold"
  echo "  2. _FLUSH_TRIGGER written to S3:"
  echo "       aws s3 ls s3://$CHECKPOINT_BUCKET/checkpoints/ --recursive | grep _FLUSH_TRIGGER"
  echo "  3. node cordoned:   kubectl get nodes   (SchedulingDisabled)"
  echo "  4. pod deleted + rescheduled onto the second Spot node"
  echo "  5. training resumes from checkpoint (job pod logs)"
  echo "  6. SQS risk event on argus-risk-events (validates the P-009 fix)"
  log "Streaming operator logs (Ctrl-C to stop):"
  kubectl logs -f deploy/${HELM_RELEASE}-operator --namespace "$NAMESPACE"
}

cmd_status() {
  log "Nodes";     kubectl get nodes -L lifecycle,workload || true
  log "Pods";      kubectl get pods -o wide --namespace "$NAMESPACE" || true
  log "SpotResilientJobs"; kubectl get spotresilientjobs --namespace "$NAMESPACE" || true
  log "Flush markers in S3"
  aws s3 ls "s3://$CHECKPOINT_BUCKET/checkpoints/" --recursive 2>/dev/null | grep _FLUSH_TRIGGER || echo "  (none)"
}

cmd_down() {
  warn "Destroying the EC2 node groups (the billed part). Control plane stays for IRSA."
  confirm "Tear down node groups now?"
  terraform -chdir="$TF_DIR" destroy \
    -target=aws_eks_node_group.spot_nodes \
    -target=aws_eks_node_group.system_nodes
  log "Node groups destroyed. Verify nothing lingers:"
  aws eks list-nodegroups --cluster-name "$CLUSTER" --region "$REGION" || true
  echo
  echo "To also drop the control plane (\$0.10/hr) and stop ALL EKS cost:"
  echo "  terraform -chdir=$TF_DIR destroy -target=aws_eks_cluster.main"
  echo "  (do this only when you no longer need IRSA / OIDC live)"
}

case "${1:-}" in
  up)     cmd_up ;;
  deploy) cmd_deploy ;;
  test)   cmd_test ;;
  status) cmd_status ;;
  down)   cmd_down ;;
  *) echo "usage: $0 {up|deploy|test|status|down}"; exit 1 ;;
esac
