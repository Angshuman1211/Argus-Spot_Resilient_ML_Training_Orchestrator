#!/usr/bin/env bash
#
# Objective 1 — survive a REAL EC2 Spot reclaim (earns the "survived Spot" claim).
# Requires a PAID-PLAN account (Free-Tier plan cannot launch Spot / ML types — P-013).
#
# Assumes the cluster + operator + predict-service + training pod are deployed
# (run ./demo/week6_runbook.sh up + deploy first, OR this script's spot-up brings
# up Spot nodes and you then run week6_runbook.sh deploy).
#
# Phases (run one at a time):
#   spot-up    apply the workload node group as real SPOT (STARTS billing)
#   nth        install AWS Node Termination Handler (drains nodes on the Spot notice)
#   fis-setup  create the FIS IAM role + experiment template (one-time)
#   fis-run    fire a REAL 2-min Spot interruption via FIS on a training node
#   evidence   print the interruption -> checkpoint -> reschedule -> resume timeline
#   down       destroy node groups + control plane (STOPS billing) <-- ALWAYS RUN
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TF_DIR="$PROJECT_ROOT/terraform"

REGION="eu-north-1"
CLUSTER="argus-eks"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argus_env.sh"
ACCOUNT="$ARGUS_ACCOUNT_ID"
NODEGROUP="argus-spot-nodes"
CKPT_BUCKET="$ARGUS_CKPT_BUCKET"
FIS_ROLE="argus-fis-spot"
FIS_TEMPLATE_NAME="argus-spot-interruption"
NTH_QUEUE="argus-nth"
NODE_ROLE="argus-eks-node-role"
EMAIL="${ALERT_EMAIL:-panchakarma.pranalogic@gmail.com}"
export AWS_PAGER=""

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
confirm() { read -r -p "$1 [y/N] " r; [[ "$r" =~ ^[Yy]$ ]] || { echo aborted; exit 1; }; }

cmd_spot_up() {
  warn "Real Spot + real billing. Paid-plan account required. Run 'down' when done."
  confirm "Bring up real SPOT workload nodes?"
  terraform -chdir="$TF_DIR" apply \
    -var="alert_email=$EMAIL" \
    -var="workload_capacity_type=SPOT" \
    -var='workload_instance_types=["m5.large","c5.xlarge","m5.xlarge"]' \
    -var="spot_desired_size=2" \
    -target=aws_eks_node_group.system_nodes \
    -target=aws_eks_node_group.spot_nodes
  aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"
  kubectl wait --for=condition=Ready nodes --all --timeout=300s
  kubectl get nodes -L lifecycle,workload,node.kubernetes.io/instance-type
  # A recreated cluster has a NEW OIDC issuer, so IRSA must be re-wired or the
  # operator + training pod can't assume their roles (no S3 checkpoints).
  log "Re-wiring IRSA for the (re)created cluster (OIDC provider + operator role)"
  terraform -chdir="$TF_DIR" apply \
    -var="alert_email=$EMAIL" \
    -target=aws_iam_openid_connect_provider.eks \
    -target=aws_iam_role.operator_irsa \
    -target=aws_iam_role_policy.operator_irsa \
    -auto-approve
  log "Spot nodes up + IRSA wired. Now: ./demo/week6_runbook.sh deploy, then apply the training pod."
}

cmd_nth() {
  log "Installing AWS Node Termination Handler (IMDS mode — drains on Spot notice)"
  helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
  helm repo update eks
  helm upgrade --install aws-node-termination-handler eks/aws-node-termination-handler \
    --namespace kube-system \
    --set enableSpotInterruptionDraining=true \
    --set enableRebalanceMonitoring=true
  kubectl -n kube-system rollout status ds/aws-node-termination-handler --timeout=120s
  log "NTH running. On a real Spot notice it will cordon+drain the node -> pods get SIGTERM."
}

cmd_nth_queue() {
  # NTH in Queue Processor mode. Works WITHOUT FIS: NTH drains a node on ANY
  # interruption event in the SQS queue — a REAL Spot warning (routed via
  # EventBridge) or one we inject (cmd_inject). Same drain->SIGTERM->resume path.
  log "1/5 SQS queue for NTH"
  QURL=$(aws sqs create-queue --queue-name "$NTH_QUEUE" --region "$REGION" \
         --attributes MessageRetentionPeriod=300 --query QueueUrl --output text 2>/dev/null) \
    || QURL=$(aws sqs get-queue-url --queue-name "$NTH_QUEUE" --region "$REGION" --query QueueUrl --output text)
  QARN=$(aws sqs get-queue-attributes --queue-url "$QURL" --region "$REGION" \
         --attribute-names QueueArn --query 'Attributes.QueueArn' --output text)
  echo "  $QURL"
  # Steps 2-3 (EventBridge -> queue) only matter for capturing NATURAL Spot
  # reclaims. Injection sends directly to the queue with your own creds, so these
  # are non-fatal — the injection demo works even if they're skipped.
  log "2/5 Allow EventBridge to feed the queue (optional; for real Spot warnings)"
  QPOLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},"Action":"sqs:SendMessage","Resource":"'"$QARN"'"}]}'
  { python3 -c "import json,sys;print(json.dumps({'Policy':sys.argv[1]}))" "$QPOLICY" > /tmp/argus_nth_attrs.json \
    && aws sqs set-queue-attributes --queue-url "$QURL" --region "$REGION" --attributes file:///tmp/argus_nth_attrs.json; } \
    || warn "queue-policy step skipped (only affects natural-interruption capture; injection still works)"
  log "3/5 EventBridge rule (optional; for real Spot warnings)"
  { aws events put-rule --name argus-spot-interruption --region "$REGION" \
      --event-pattern '{"source":["aws.ec2"],"detail-type":["EC2 Spot Instance Interruption Warning"]}' >/dev/null \
    && aws events put-targets --rule argus-spot-interruption --region "$REGION" --targets "Id=nthq,Arn=$QARN" >/dev/null; } \
    || warn "eventbridge wiring skipped (natural-capture only; injection still works)"
  # NTH gets creds via IRSA, NOT the node role: EKS managed nodes cap the IMDS
  # hop limit so pods can't reach the instance metadata endpoint (NoCredentialProviders).
  log "4/5 NTH IRSA role (trust: kube-system:aws-node-termination-handler)"
  local ISSUER OIDC_ARN NTH_ROLE_ARN
  ISSUER=$(aws eks describe-cluster --name "$CLUSTER" --region "$REGION" --query 'cluster.identity.oidc.issuer' --output text | sed 's#https://##')
  OIDC_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/${ISSUER}"
  NTH_ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/argus-nth-irsa"
  cat > /tmp/argus_nth_trust.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Federated":"$OIDC_ARN"},"Action":"sts:AssumeRoleWithWebIdentity","Condition":{"StringEquals":{"${ISSUER}:sub":"system:serviceaccount:kube-system:aws-node-termination-handler","${ISSUER}:aud":"sts.amazonaws.com"}}}]}
EOF
  aws iam create-role --role-name argus-nth-irsa --assume-role-policy-document file:///tmp/argus_nth_trust.json >/dev/null 2>&1 \
    || aws iam update-assume-role-policy --role-name argus-nth-irsa --policy-document file:///tmp/argus_nth_trust.json
  aws iam put-role-policy --role-name argus-nth-irsa --policy-name nth \
    --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["sqs:ReceiveMessage","sqs:DeleteMessage","sqs:GetQueueAttributes","ec2:DescribeInstances","autoscaling:DescribeAutoScalingInstances","autoscaling:CompleteLifecycleAction"],"Resource":"*"}]}'
  log "5/5 Install NTH (queue mode) with the IRSA-annotated SA, then restart to inject the token"
  helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
  helm repo update eks >/dev/null
  cat > /tmp/argus_nth_values.yaml <<EOF
enableSqsTerminationDraining: true
queueURL: $QURL
awsRegion: $REGION
# EKS nodes aren't tagged aws-node-termination-handler/managed, so without this
# NTH silently skips the interruption event. Disable the tag gate.
checkTagBeforeDraining: false
checkASGTagBeforeDraining: false
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: $NTH_ROLE_ARN
EOF
  helm upgrade --install aws-node-termination-handler eks/aws-node-termination-handler \
    --namespace kube-system -f /tmp/argus_nth_values.yaml
  kubectl -n kube-system rollout restart deploy/aws-node-termination-handler >/dev/null 2>&1 || true
  kubectl -n kube-system rollout status deploy/aws-node-termination-handler --timeout=120s
  log "NTH queue mode up (IRSA). Inject with: ./scripts/objective1_real_spot.sh inject"
}

cmd_inject() {
  # Send a schema-correct EC2 Spot Interruption Warning for the node running the
  # training pod. NTH cannot distinguish it from a real EventBridge-delivered one
  # -> it cordons + drains that node, exercising the real reactive path.
  QURL=$(aws sqs get-queue-url --queue-name "$NTH_QUEUE" --region "$REGION" --query QueueUrl --output text 2>/dev/null) \
    || { echo "No NTH queue — run nth-queue first."; exit 1; }
  NODE=$(kubectl get pod cifar10-test -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  [ -z "$NODE" ] && { echo "cifar10-test is not scheduled — deploy the training pod first."; exit 1; }
  IID=$(kubectl get node "$NODE" -o jsonpath='{.spec.providerID}' 2>/dev/null | sed 's#.*/##')
  [ -z "$IID" ] && { echo "Could not resolve instance id for node $NODE"; exit 1; }
  echo "  training pod on node $NODE  (instance $IID)"
  warn "Injecting a schema-correct EC2 Spot Interruption Warning for $IID."
  echo "  (Honest framing: this exercises the interruption HANDLER via a real-schema"
  echo "   event; the instance is not AWS-reclaimed. Say so precisely in the paper.)"
  confirm "Inject?"
  BODY="{\"version\":\"0\",\"id\":\"$(uuidgen)\",\"detail-type\":\"EC2 Spot Instance Interruption Warning\",\"source\":\"aws.ec2\",\"account\":\"$ACCOUNT\",\"time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"region\":\"$REGION\",\"resources\":[\"arn:aws:ec2:$REGION:$ACCOUNT:instance/$IID\"],\"detail\":{\"instance-id\":\"$IID\",\"instance-action\":\"terminate\"}}"
  MID=$(aws sqs send-message --queue-url "$QURL" --region "$REGION" --message-body "$BODY" --query MessageId --output text)
  log "Injected (msg $MID). NTH should cordon+drain $NODE within ~30s."
  echo "  Watch: ./scripts/objective1_real_spot.sh evidence"
}

cmd_check_fis() {
  if aws fis list-experiment-templates --region "$REGION" >/dev/null 2>&1; then
    echo "✅ FIS ACTIVE in $REGION — Objective 1 can proceed"
  else
    echo "❌ FIS not active yet (SubscriptionRequiredException — Free->Paid upgrade still propagating). Retry later."
  fi
}

cmd_fis_setup() {
  # Preflight: FIS needs the account subscribed/activated. After a Free->Paid
  # upgrade this can lag behind EC2 by minutes-hours. Fail fast, not after EKS is up.
  if ! aws fis list-experiment-templates --region "$REGION" >/dev/null 2>&1; then
    warn "FIS is not active on this account/region yet (SubscriptionRequiredException)."
    echo "  Likely the Free->Paid upgrade is still propagating. Check with:"
    echo "    ./scripts/objective1_real_spot.sh check-fis"
    echo "  Re-run fis-setup once it reports ACTIVE."
    exit 1
  fi
  log "Creating FIS IAM role '$FIS_ROLE' (assumed by fis.amazonaws.com)"
  aws iam create-role --role-name "$FIS_ROLE" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"fis.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    2>/dev/null || echo "  (role exists)"
  aws iam put-role-policy --role-name "$FIS_ROLE" --policy-name spot-interrupt \
    --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ec2:SendSpotInstanceInterruptions","ec2:DescribeInstances"],"Resource":"*"}]}'
  ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${FIS_ROLE}"

  log "Creating FIS experiment template '$FIS_TEMPLATE_NAME'"
  cat > /tmp/argus_fis_template.json <<JSON
{
  "description": "Send a real Spot interruption to an Argus training node",
  "roleArn": "${ROLE_ARN}",
  "stopConditions": [{"source": "none"}],
  "targets": {
    "SpotNodes": {
      "resourceType": "aws:ec2:spot-instance",
      "resourceTags": {"eks:nodegroup-name": "${NODEGROUP}"},
      "selectionMode": "COUNT(1)"
    }
  },
  "actions": {
    "interrupt": {
      "actionId": "aws:ec2:send-spot-instance-interruptions",
      "parameters": {"durationBeforeInterruption": "PT2M"},
      "targets": {"SpotInstances": "SpotNodes"}
    }
  },
  "tags": {"project": "argus"}
}
JSON
  TID=$(aws fis create-experiment-template --region "$REGION" \
        --cli-input-json file:///tmp/argus_fis_template.json \
        --query 'experimentTemplate.id' --output text 2>/tmp/argus_fis_err) || {
        echo "  create-experiment-template FAILED:"; sed 's/^/    /' /tmp/argus_fis_err
        echo "  (if a template already exists, reuse it: aws fis list-experiment-templates --region $REGION)"; exit 1; }
  echo "  Template id: $TID  (fis-run picks it up automatically)"
}

cmd_fis_run() {
  TID=$(aws fis list-experiment-templates --region "$REGION" \
        --query "experimentTemplates[?tags.project=='argus'].id | [0]" --output text 2>/dev/null)
  [ -z "$TID" ] || [ "$TID" = "None" ] && { echo "No template — run fis-setup first."; exit 1; }
  warn "This sends a REAL 2-minute Spot interruption to a training node."
  confirm "Fire the interruption (template $TID)?"
  log "Recording checkpoint state BEFORE"
  aws s3 ls "s3://${CKPT_BUCKET}/checkpoints/cifar10-test/" 2>&1 || true
  aws fis start-experiment --region "$REGION" --experiment-template-id "$TID" \
    --query 'experiment.{id:id,state:state.status}' --output json
  log "Interruption sent. Watch: ./scripts/objective1_real_spot.sh evidence"
}

cmd_evidence() {
  log "Nodes (watch for the interrupted one going NotReady/SchedulingDisabled)"
  kubectl get nodes -L workload 2>&1 || true
  log "NTH log (should show 'Spot interruption' + cordon/drain)"
  { kubectl -n kube-system logs deploy/aws-node-termination-handler --tail=25 2>/dev/null; \
    kubectl -n kube-system logs ds/aws-node-termination-handler --tail=25 2>/dev/null; } \
    | grep -iE 'spot|cordon|drain|interrupt|taint' || echo "  (nothing yet)"
  log "Checkpoint in S3 (should update at interruption time)"
  aws s3 ls "s3://${CKPT_BUCKET}/checkpoints/cifar10-test/" 2>&1 || true
  log "Training pod (should reschedule to another node, then resume from checkpoint)"
  kubectl get pod cifar10-test -o wide 2>&1 || echo "  (gone/rescheduling)"
  kubectl logs cifar10-test --tail=8 2>&1 | grep -iaE 'resum|checkpoint|epoch [0-9]' | grep -viE '[0-9]\.[0-9]%' || echo "  (no resume log yet)"
}

cmd_down() {
  warn "Destroying node groups + control plane (the billed part)."
  confirm "Tear down now?"
  terraform -chdir="$TF_DIR" destroy \
    -var="alert_email=$EMAIL" \
    -target=aws_eks_node_group.spot_nodes \
    -target=aws_eks_node_group.system_nodes \
    -target=aws_eks_cluster.main
  log "Verify \$0:  ./scripts/verify_teardown.sh"
}

case "${1:-}" in
  spot-up)   cmd_spot_up ;;
  nth)       cmd_nth ;;
  nth-queue) cmd_nth_queue ;;
  inject)    cmd_inject ;;
  check-fis) cmd_check_fis ;;
  fis-setup) cmd_fis_setup ;;
  fis-run)   cmd_fis_run ;;
  evidence)  cmd_evidence ;;
  down)      cmd_down ;;
  *) cat <<EOF
usage: $0 <cmd>
  spot-up     apply real SPOT workload nodes (billing starts)
  nth-queue   NTH Queue Processor mode — works WITHOUT FIS (Option A)
  inject      send a schema-correct Spot Interruption Warning at the training node
  evidence    show interruption -> checkpoint -> reschedule -> resume
  down        destroy node groups + control plane (ALWAYS run)
  nth         NTH IMDS mode (used with the FIS path)
  check-fis   is AWS FIS active yet? (blocked by account plan, P-019)
  fis-setup   create FIS role + template (needs FIS active)
  fis-run     fire a REAL AWS Spot interruption via FIS
EOF
     exit 1 ;;
esac
