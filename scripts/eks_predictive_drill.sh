#!/usr/bin/env bash
# Real-EKS drill of the PREDICTIVE path: real operator, retrained model served live, training pods with large
# checkpoints. For each job it records how long the flush takes, whether the operator waited for it, and how long
# until the replacement pod resumes.
#
#   ./scripts/eks_predictive_drill.sh prepare   # roll operator + predict service to the v2 images, save old state
#   ./scripts/eks_predictive_drill.sh run NAME EXTRA_MB THRESHOLD   # one drill job
#   ./scripts/eks_predictive_drill.sh restore   # put the previous deployments back
#
# The live price feed needs scripts/price_feeder.py running somewhere with EC2 credentials, because the pods'
# IAM role can read only the checkpoints bucket (no EC2 or Lambda-bucket access, and we cannot grant any).
set -euo pipefail
CTX="${KUBE_CONTEXT:-argus-eks}"
K="kubectl --context $CTX"
STATE="${STATE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/demo/drill/saved_state}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argus_env.sh"
REG="$ARGUS_REGISTRY"
OUT="${DRILL_OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/demo/drill/results}"
mkdir -p "$STATE" "$OUT"

cmd_prepare() {
  $K get deploy argus-operator argus-predict-service -n default -o yaml > "$STATE/deployments_before.yaml"
  $K get srj -A -o yaml > "$STATE/srj_before.yaml" || true
  echo "saved previous state to $STATE"
  $K set image deploy/argus-predict-service predict-service=$REG/argus/predict-service:v2-xgb -n default
  $K set env deploy/argus-predict-service PRICE_SOURCE=s3object PRICE_S3_URI=s3://$ARGUS_CKPT_BUCKET/prices/latest.csv AWS_REGION=eu-north-1 -n default
  $K patch deploy argus-predict-service -n default -p '{"spec":{"template":{"spec":{"serviceAccountName":"argus-operator"}}}}'
  $K set image deploy/argus-operator operator=$REG/argus/operator:v2 -n default || \
    $K set image deploy/argus-operator '*'=$REG/argus/operator:v2 -n default
  $K rollout status deploy/argus-predict-service -n default --timeout=300s
  $K rollout status deploy/argus-operator -n default --timeout=300s
}

cmd_run() {
  export NAME="$1" EXTRA_MB="$2" THRESHOLD="$3" EPOCHS="${EPOCHS:-3}"
  local log="$OUT/$NAME.log"
  : > "$log"
  local d; d="$(dirname "${BASH_SOURCE[0]}")/../demo/drill"
  # pod first, CR second: if the CR existed before the pod was training, the operator's first poll would fire
  # its trigger at nothing
  envsubst < "$d/pod.yaml.tmpl" | $K apply -f - | tee -a "$log"
  $K wait --for=condition=Ready pod/$NAME --timeout=900s
  sleep 45   # let the training loop start (it polls the flush trigger every 100 batches)
  echo "pod ready $(date -u +%T)" | tee -a "$log"
  envsubst < "$d/cr.yaml.tmpl" | $K apply -f - | tee -a "$log"
  echo "CR created $(date -u +%T)" | tee -a "$log"
  # wait until the operator deletes the pod (it triggers, waits for the flush, cordons, deletes)
  local t0; t0=$(date +%s)
  while $K get pod $NAME -o name >/dev/null 2>&1; do
    sleep 2
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timed out waiting for migration" | tee -a "$log"; return 1; }
  done
  echo "pod deleted by operator after $(( $(date +%s) - t0 )) s" | tee -a "$log"
  # the operator deletes a bare pod; recreate it as a controller would (the CR already exists)
  envsubst < "$d/pod.yaml.tmpl" | $K apply -f - | tee -a "$log"
  $K wait --for=condition=Ready pod/$NAME --timeout=900s
  echo "replacement pod ready $(date -u +%T)" | tee -a "$log"
  sleep 20
  $K logs pod/$NAME -c train | tee "$OUT/$NAME.train.log" | grep -E "Resuming|Loading|Saving|uploaded" | tee -a "$log"
  $K logs deploy/argus-operator -n default --since=30m | grep -E "$NAME" | tee "$OUT/$NAME.operator.log" | tail -25 | tee -a "$log"
  # let the operator uncordon on its own; show cordon state
  $K get nodes | tee -a "$log"
}

cmd_start() {
  # start a job and leave it running (used for the job that should NOT migrate)
  export NAME="$1" EXTRA_MB="$2" THRESHOLD="$3" EPOCHS="${EPOCHS:-3}"
  local d; d="$(dirname "${BASH_SOURCE[0]}")/../demo/drill"
  envsubst < "$d/pod.yaml.tmpl" | $K apply -f -
  $K wait --for=condition=Ready pod/$NAME --timeout=900s
  sleep 45
  envsubst < "$d/cr.yaml.tmpl" | $K apply -f -
}

cmd_restore() {
  $K apply -f "$STATE/deployments_before.yaml"
  $K get nodes -o name | while read -r n; do $K uncordon "${n#node/}" || true; done
}

case "${1:-}" in
  prepare) cmd_prepare ;;
  run) shift; cmd_run "$@" ;;
  start) shift; cmd_start "$@" ;;
  restore) cmd_restore ;;
  *) echo "usage: $0 prepare | run NAME EXTRA_MB THRESHOLD | start NAME EXTRA_MB THRESHOLD | restore"; exit 1 ;;
esac
