# Account-specific names for the shell scripts, from one place. Source it:  . "$SCRIPT_DIR/argus_env.sh"
#
# The AWS account comes from ARGUS_ACCOUNT_ID if set, otherwise from whoever the AWS CLI is logged in as,
# so moving to a new account needs no edits. Manifests under demo/ and k8s/ carry ${ARGUS_ACCOUNT_ID}
# placeholders: apply them with  envsubst < file.yaml | kubectl apply -f -
: "${AWS_REGION:=eu-north-1}"
if [ -z "${ARGUS_ACCOUNT_ID:-}" ]; then
  ARGUS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" || {
    echo "argus_env.sh: set ARGUS_ACCOUNT_ID or log in with the AWS CLI" >&2; exit 1; }
fi
export ARGUS_ACCOUNT_ID AWS_REGION
export ARGUS_REGISTRY="$ARGUS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
export ARGUS_CKPT_BUCKET="${ARGUS_CKPT_BUCKET:-argus-checkpoints-$ARGUS_ACCOUNT_ID}"
