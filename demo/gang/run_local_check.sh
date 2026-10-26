#!/usr/bin/env bash
# Local check of gang mode against a real Kubernetes API server (minikube, 3 nodes). No AWS.
#   demo/gang/run_local_check.sh            # needs Docker, minikube, kubectl, Python with kopf and moto[server]
#   KEEP=1 demo/gang/run_local_check.sh     # keep the minikube profile afterwards
# Evidence goes to demo/gang/results/. PY is a Python with kopf, kubernetes, httpx, fastapi, uvicorn, boto3;
# MOTO_PATH (optional) is a directory to add to PYTHONPATH that holds moto[server].
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
PROFILE=argus-gang
PY=${PY:-python}
OUT=demo/gang/results
mkdir -p "$OUT"
S3_PORT=${S3_PORT:-4566}
PRED_PORT=${PRED_PORT:-8011}
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=eu-north-1

minikube status -p $PROFILE >/dev/null 2>&1 || minikube start -p $PROFILE --nodes 3 --cpus 2 --memory 2048 --driver docker
kubectl config use-context $PROFILE >/dev/null
nodes=($(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'))
types=(t-a t-b t-c)
for i in 0 1 2; do
  kubectl label node "${nodes[$i]}" node.kubernetes.io/instance-type=${types[$i]} topology.kubernetes.io/zone=local-a --overwrite >/dev/null
  kubectl uncordon "${nodes[$i]}" >/dev/null
done
RISKY=${nodes[1]}   # instance type t-b

kubectl apply -f operator/crd/spotresilientjob.yaml >/dev/null
kubectl delete srj gangtest --ignore-not-found >/dev/null
kubectl delete pod -l argus.io/job=gangtest --ignore-not-found --wait >/dev/null
kubectl apply -f demo/gang/gang-pods.yaml >/dev/null
kubectl wait --for=condition=Ready pod -l argus.io/job=gangtest --timeout=180s >/dev/null
# make sure a member sits on the risky node (anti-affinity is only preferred)
if ! kubectl get pods -l argus.io/job=gangtest -o jsonpath='{.items[*].spec.nodeName}' | grep -qw "$RISKY"; then
  echo "no member landed on the risky node $RISKY; rerun" >&2; exit 1
fi
kubectl get pods -l argus.io/job=gangtest -o wide > "$OUT/before_pods.txt"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; [ "${KEEP:-0}" = 1 ] || minikube delete -p $PROFILE >/dev/null 2>&1 || true; }
trap cleanup EXIT

PYTHONPATH="${MOTO_PATH:-}" "$PY" -m moto.server -p $S3_PORT > "$OUT/moto.log" 2>&1 & pids+=($!)
(cd operator && MOCK_RISKY_TYPES=t-b MOCK_RISK_SCORE=0.9 "$PY" -m uvicorn controller.mock_predict:app --port $PRED_PORT > "$ROOT/$OUT/predict.log" 2>&1) & pids+=($!)
sleep 6
AWS_ENDPOINT_URL=http://127.0.0.1:$S3_PORT "$PY" -c "import boto3; boto3.client('s3').create_bucket(Bucket='argus-local', CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})"

(cd operator && PREDICT_SERVICE_URL=http://127.0.0.1:$PRED_PORT AWS_ENDPOINT_URL=http://127.0.0.1:$S3_PORT \
  POLL_INTERVAL_SECONDS=15 CHECKPOINT_WAIT_SECONDS=10 RECREATE_WAIT_SECONDS=60 METRICS_PORT=18080 \
  "$PY" -m controller.main > "$ROOT/$OUT/operator.log" 2>&1) & pids+=($!)
sleep 8
kubectl apply -f demo/gang/gang-job.yaml >/dev/null

# wait for the move, then for the uncordon once every member runs elsewhere
for _ in $(seq 60); do grep -q "moved all" "$OUT/operator.log" && break; sleep 5; done
kubectl get nodes > "$OUT/during_nodes.txt"
kubectl wait --for=condition=Ready pod -l argus.io/job=gangtest --timeout=180s >/dev/null
for _ in $(seq 40); do [ -z "$(kubectl get node "$RISKY" -o jsonpath='{.spec.unschedulable}')" ] && break; sleep 5; done
kubectl get pods -l argus.io/job=gangtest -o wide > "$OUT/after_pods.txt"
kubectl get nodes > "$OUT/after_nodes.txt"
kubectl get srj gangtest -o yaml > "$OUT/srj_status.yaml"

# checks
fail=0
n_trig=$(grep -c "Trigger .* written" "$OUT/operator.log" || true)
[ "$n_trig" = 1 ] && echo "ok   one flush request for the whole job" || { echo "FAIL flush requests: $n_trig"; fail=1; }
grep -q "SchedulingDisabled" <(grep "$RISKY" "$OUT/during_nodes.txt") && echo "ok   risky node $RISKY was cordoned" || { echo "FAIL risky node not cordoned"; fail=1; }
n_cordoned=$(grep -c "SchedulingDisabled" "$OUT/during_nodes.txt" || true)
[ "$n_cordoned" = 1 ] && echo "ok   only one node cordoned" || { echo "FAIL cordoned nodes: $n_cordoned"; fail=1; }
n_run=$(grep -c " Running " "$OUT/after_pods.txt" || true)
[ "$n_run" = 3 ] && echo "ok   all three members running again" || { echo "FAIL running members: $n_run"; fail=1; }
before=$(awk 'NR>1{print $1, $6}' "$OUT/before_pods.txt" | sort); after=$(awk 'NR>1{print $1, $6}' "$OUT/after_pods.txt" | sort)
grep -qw "$RISKY" <(awk 'NR>1{print $7}' "$OUT/after_pods.txt") && { echo "FAIL a member came back on the risky node"; fail=1; } || echo "ok   no member on the risky node"
[ "$before" != "$after" ] && echo "ok   every member was recreated (new pod IPs)" || { echo "FAIL pods unchanged"; fail=1; }
[ -z "$(kubectl get node "$RISKY" -o jsonpath='{.spec.unschedulable}')" ] && echo "ok   risky node uncordoned after the move" || { echo "FAIL still cordoned"; fail=1; }
exit $fail
